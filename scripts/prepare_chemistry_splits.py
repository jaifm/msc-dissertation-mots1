"""Prepare standardised Composite_Mix_ID-based train/test splits.

CHEMISTRY DEFINITION
--------------------
A chemistry is a unique combination of Formulation + Dispersant + Solid
Content + Solid Additive, encoded in ``Composite_Mix_ID`` (e.g.
``F1_Triton_73.0_0.5``).  This gives 67 unique chemistries across the
dataset.  The previous coarser definition (Formulation × Dispersant only,
12 groups) is no longer used.

WHAT THIS PRODUCES
------------------
Three families of splits, all keyed on ``Composite_Mix_ID``:

  1. ``default/`` -- one fixed split that holds out a stratified ~25 % sample
     of Composite_Mix_IDs (sampled within each Formulation × Dispersant group
     so every dispersant family remains represented in training).  Groups with
     ≤ 2 IDs contribute no test rows to preserve training coverage.

  2. ``loo/<id>/`` -- 67 leave-one-out splits, one per Composite_Mix_ID.
     All batches of the held-out ID form the test set; all other IDs train.

  3. ``few_shot/<id>_k<k>/`` -- for Composite_Mix_IDs with ≥ 3 rows, leak
     k ∈ {1, 2} batches back into training and test on the remainder.
     Models how quickly prediction recovers once even one measurement of a
     new recipe is available.

  4. ``batch_b1_to_b2/`` and ``batch_b2_to_b1/`` -- cross-batch
     generalisation splits.  Batch_1 and Batch_2 are almost entirely
     disjoint recipe sets (18 B1-only, 37 B2-only, 12 shared); the shared
     recipes go into training only so the test sets contain genuinely
     unseen Composite_Mix_IDs in both directions.

OUTPUT STRUCTURE
----------------
::

    data/processed/splits_chemistry/
    ├── MANIFEST.json
    ├── default/
    │   ├── train.csv
    │   └── test.csv
    ├── loo/
    │   ├── F1_Hypermer_69.0_0.0/{train,test}.csv
    │   └── ... (67 total)
    └── few_shot/
        ├── F1_Hypermer_69.0_0.25_k1/{train,test}.csv
        └── ...

USAGE
-----
Run once from the project root::

    python scripts/prepare_chemistry_splits.py

The fixed splits are immutable once generated — their SHA-256 hashes in
MANIFEST.json are the reproducibility contract for every downstream
experiment.  ``--batch-only`` adds/refreshes only the two cross-batch
splits (and their manifest entries) without touching the default, LOO, or
few-shot families::

    python scripts/prepare_chemistry_splits.py --batch-only
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fraction of IDs to hold out per Formulation × Dispersant group (default split).
DEFAULT_HOLDOUT_FRACTION: float = 0.25

# Only hold out from groups that have at least this many unique IDs.
# Groups with fewer IDs are kept entirely in training to preserve coverage.
MIN_IDS_TO_SAMPLE_FROM: int = 3

# Few-shot k values (number of batches leaked back into training).
FEW_SHOT_K_VALUES: Tuple[int, ...] = (1, 2)

# Minimum rows for a Composite_Mix_ID to be eligible for few-shot.
FEW_SHOT_MIN_ROWS: int = 3

SEED: int = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_TRAIN = PROJECT_ROOT / "data" / "processed" / "splits" / "expanded_train.csv"
INPUT_TEST  = PROJECT_ROOT / "data" / "processed" / "splits" / "expanded_test.csv"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "splits_chemistry"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_full_dataset() -> pd.DataFrame:
    """Concatenate existing train/test CSVs and clean blank rows."""
    if not INPUT_TRAIN.exists() or not INPUT_TEST.exists():
        raise FileNotFoundError(
            f"Missing input splits:\n  {INPUT_TRAIN}\n  {INPUT_TEST}"
        )
    df = pd.concat(
        [pd.read_csv(INPUT_TRAIN), pd.read_csv(INPUT_TEST)],
        axis=0, ignore_index=True,
    )
    before = len(df)
    df = df[df["Formulation"].isin(["F1", "F2", "F3"])].copy()
    df = df.dropna(subset=["NMC_pct", "C65_pct", "KS6L_pct", "PVDF_pct"])
    if before != len(df):
        print(f"  dropped {before - len(df)} blank/NaN row(s) ({before} → {len(df)})")
    return df.reset_index(drop=True)


def _write_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.csv"
    test_path  = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return {
        "train_sha256": _sha256(train_path),
        "test_sha256":  _sha256(test_path),
    }


def _manifest_entry(
    name: str,
    purpose: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    hashes: Dict[str, str],
) -> Dict[str, object]:
    return {
        "name": name,
        "purpose": purpose,
        "n_train": int(len(train_df)),
        "n_test":  int(len(test_df)),
        "train_sha256": hashes["train_sha256"],
        "test_sha256":  hashes["test_sha256"],
        "ids_in_train": sorted(train_df["Composite_Mix_ID"].unique().tolist()),
        "ids_in_test":  sorted(test_df["Composite_Mix_ID"].unique().tolist()),
    }


# ---------------------------------------------------------------------------
# Split generators
# ---------------------------------------------------------------------------


def _select_default_held_out_ids(
    df: pd.DataFrame,
    fraction: float,
    min_ids: int,
    rng: np.random.Generator,
) -> List[str]:
    """Stratified sample of Composite_Mix_IDs to hold out for the default split.

    Within each Formulation × Dispersant group, holds out ``ceil(n * fraction)``
    IDs chosen without replacement.  Groups with fewer than ``min_ids`` unique
    IDs contribute nothing to the test set (preserving coverage).
    """
    held_out: List[str] = []
    df["_fd"] = df["Formulation"] + "_" + df["Dispersent_Type"]
    for fd_group, sub in df.groupby("_fd"):
        ids_in_group = sorted(sub["Composite_Mix_ID"].unique().tolist())
        if len(ids_in_group) < min_ids:
            continue
        n_hold = max(1, round(len(ids_in_group) * fraction))
        chosen = rng.choice(ids_in_group, size=n_hold, replace=False).tolist()
        held_out.extend(chosen)
    df.drop(columns=["_fd"], inplace=True)
    return sorted(held_out)


def _build_default_split(
    df: pd.DataFrame,
    held_out_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["Composite_Mix_ID"].isin(set(held_out_ids))
    return (
        df.loc[~mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )


def _build_loo_split(
    df: pd.DataFrame,
    mix_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["Composite_Mix_ID"] == mix_id
    return (
        df.loc[~mask].reset_index(drop=True),
        df.loc[mask].reset_index(drop=True),
    )


def _build_few_shot_split(
    df: pd.DataFrame,
    mix_id: str,
    k: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leak ``k`` rows of ``mix_id`` back into training."""
    sub = df[df["Composite_Mix_ID"] == mix_id]
    if len(sub) <= k:
        raise ValueError(
            f"few-shot k={k} >= rows ({len(sub)}) for {mix_id!r}"
        )
    leaked_idx = set(int(i) for i in rng.choice(sub.index.to_numpy(), size=k, replace=False))
    test_mask = (df["Composite_Mix_ID"] == mix_id) & ~df.index.isin(leaked_idx)
    return (
        df.loc[~test_mask].reset_index(drop=True),
        df.loc[test_mask].reset_index(drop=True),
    )


def _build_batch_splits(
    df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Cross-batch splits with shared recipes confined to training."""
    batch_sets = df.groupby("Composite_Mix_ID")["Source_Batch"].apply(set)
    only_b1 = set(batch_sets[batch_sets.apply(lambda s: s == {"Batch_1"})].index)
    only_b2 = set(batch_sets[batch_sets.apply(lambda s: s == {"Batch_2"})].index)
    both = set(batch_sets[batch_sets.apply(lambda s: s == {"Batch_1", "Batch_2"})].index)
    print(f"  Batch_1-only recipes : {len(only_b1)}")
    print(f"  Batch_2-only recipes : {len(only_b2)}")
    print(f"  Both-batch recipes   : {len(both)}")

    b1_train = df[df["Source_Batch"] == "Batch_1"].reset_index(drop=True)
    b1_test = df[df["Composite_Mix_ID"].isin(only_b2)].reset_index(drop=True)
    b2_train = df[df["Source_Batch"] == "Batch_2"].reset_index(drop=True)
    b2_test = df[df["Composite_Mix_ID"].isin(only_b1)].reset_index(drop=True)
    for train_df, test_df in [(b1_train, b1_test), (b2_train, b2_test)]:
        overlap = (set(train_df["Composite_Mix_ID"].unique())
                   & set(test_df["Composite_Mix_ID"].unique()))
        assert not overlap, f"cross-batch leakage: {sorted(overlap)[:3]}"
    return {
        "batch_b1_to_b2": (b1_train, b1_test),
        "batch_b2_to_b1": (b2_train, b2_test),
    }


def _write_batch_splits(df: pd.DataFrame, manifest: List[Dict[str, object]]) -> None:
    print("Cross-batch splits —")
    for name, (train_df, test_df) in _build_batch_splits(df).items():
        hashes = _write_split(train_df, test_df, OUTPUT_ROOT / name)
        direction = ("Batch_1 → Batch_2" if name == "batch_b1_to_b2"
                     else "Batch_2 → Batch_1")
        manifest.append(_manifest_entry(
            name=name,
            purpose=(
                f"Cross-batch generalisation ({direction}): the source batch "
                f"trains (shared recipes included), the target batch's "
                f"exclusive recipes test. No Composite_Mix_ID overlap."
            ),
            train_df=train_df, test_df=test_df, hashes=hashes,
        ))
        print(f"  {name}: train={len(train_df)} rows "
              f"({train_df.Composite_Mix_ID.nunique()} IDs)  "
              f"test={len(test_df)} rows ({test_df.Composite_Mix_ID.nunique()} IDs)")


def batch_only() -> int:
    """Add/refresh only the cross-batch splits + their manifest entries.

    The default/LOO/few-shot families and their manifest records are left
    byte-for-byte untouched, preserving the reproducibility contract.
    """
    manifest_path = OUTPUT_ROOT / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — run the full script first."
        )
    manifest_obj = json.loads(manifest_path.read_text())
    kept = [e for e in manifest_obj["splits"]
            if not str(e["name"]).startswith("batch_")]

    df = _load_full_dataset()
    print(f"Loaded {len(df)} rows, {df['Composite_Mix_ID'].nunique()} unique Composite_Mix_IDs\n")
    new_entries: List[Dict[str, object]] = []
    _write_batch_splits(df, new_entries)

    manifest_obj["splits"] = kept + new_entries
    manifest_path.write_text(json.dumps(manifest_obj, indent=2))
    print(f"\nUpdated MANIFEST.json with {len(new_entries)} batch split entries")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Preparing Composite_Mix_ID-based train/test splits")
    print("=" * 72)
    print(f"  Output: {OUTPUT_ROOT}")
    print()

    df = _load_full_dataset()
    mix_ids = sorted(df["Composite_Mix_ID"].unique().tolist())
    print(f"Loaded {len(df)} rows, {len(mix_ids)} unique Composite_Mix_IDs")

    sizes = df.groupby("Composite_Mix_ID").size().sort_index()
    print("\nRows per Composite_Mix_ID:")
    for mid, n in sizes.items():
        fd = df[df["Composite_Mix_ID"] == mid].iloc[0]
        print(f"  {mid:<35s}  {n:2d} rows")
    print()

    if OUTPUT_ROOT.exists():
        print(f"Removing existing {OUTPUT_ROOT} (clean rebuild)")
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    manifest: List[Dict[str, object]] = []

    # 1. Default split.
    held_out_ids = _select_default_held_out_ids(
        df.copy(), DEFAULT_HOLDOUT_FRACTION, MIN_IDS_TO_SAMPLE_FROM, rng
    )
    print(f"Default split — holding out {len(held_out_ids)} IDs (~{len(held_out_ids)/len(mix_ids)*100:.0f}%):")
    for mid in held_out_ids:
        print(f"  {mid}")
    train_df, test_df = _build_default_split(df, held_out_ids)
    hashes = _write_split(train_df, test_df, OUTPUT_ROOT / "default")
    manifest.append(_manifest_entry(
        name="default",
        purpose=(
            f"Headline fixed split. Holds out {len(held_out_ids)} Composite_Mix_IDs "
            f"(~{len(held_out_ids)/len(mix_ids)*100:.0f}%), sampled proportionally "
            f"within each Formulation×Dispersant group. Tests generalisation to "
            f"unseen specific recipes."
        ),
        train_df=train_df, test_df=test_df, hashes=hashes,
    ))
    print(f"  → train={len(train_df)}, test={len(test_df)}\n")

    # 2. Leave-one-out (67 splits).
    print(f"LOO splits ({len(mix_ids)} total) —")
    for mix_id in mix_ids:
        train_df, test_df = _build_loo_split(df, mix_id)
        safe_name = mix_id.replace(" ", "_")
        hashes = _write_split(train_df, test_df, OUTPUT_ROOT / "loo" / safe_name)
        manifest.append(_manifest_entry(
            name=f"loo/{safe_name}",
            purpose=(
                f"Leave-one-out: all batches of {mix_id!r} held out for testing; "
                f"all other {len(mix_ids)-1} Composite_Mix_IDs used for training."
            ),
            train_df=train_df, test_df=test_df, hashes=hashes,
        ))
        print(f"  {mix_id:<38s} train={len(train_df):3d}  test={len(test_df):2d}")

    # 3. Few-shot splits.
    eligible = [mid for mid in mix_ids if sizes[mid] >= FEW_SHOT_MIN_ROWS]
    print(f"\nFew-shot splits (k ∈ {list(FEW_SHOT_K_VALUES)}, "
          f"{len(eligible)} IDs with ≥ {FEW_SHOT_MIN_ROWS} rows) —")
    for mix_id in eligible:
        for k in FEW_SHOT_K_VALUES:
            if sizes[mix_id] <= k:
                continue
            safe_name = mix_id.replace(" ", "_")
            name = f"few_shot/{safe_name}_k{k}"
            train_df, test_df = _build_few_shot_split(df, mix_id, k, rng)
            hashes = _write_split(
                train_df, test_df,
                OUTPUT_ROOT / "few_shot" / f"{safe_name}_k{k}",
            )
            manifest.append(_manifest_entry(
                name=name,
                purpose=(
                    f"Few-shot: {k} batch(es) of {mix_id!r} leaked into training; "
                    f"remaining {sizes[mix_id]-k} batch(es) form the test set."
                ),
                train_df=train_df, test_df=test_df, hashes=hashes,
            ))
            print(f"  {name:<50s} train={len(train_df):3d}  test={len(test_df):2d}")

    # 4. Cross-batch splits.
    print()
    _write_batch_splits(df, manifest)

    # 5. Manifest.
    manifest_obj = {
        "generated_by": "scripts/prepare_chemistry_splits.py",
        "chemistry_definition": "Composite_Mix_ID (Formulation + Dispersant + Solid_Content_pct + Solid_Additive_pct)",
        "seed": SEED,
        "n_chemistries": len(mix_ids),
        "chemistries": mix_ids,
        "default_held_out_ids": held_out_ids,
        "few_shot_k_values": list(FEW_SHOT_K_VALUES),
        "few_shot_min_rows": FEW_SHOT_MIN_ROWS,
        "splits": manifest,
    }
    (OUTPUT_ROOT / "MANIFEST.json").write_text(json.dumps(manifest_obj, indent=2))

    print(f"\nWrote {len(manifest)} splits + MANIFEST.json → {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    if "--batch-only" in sys.argv[1:]:
        sys.exit(batch_only())
    sys.exit(main())
