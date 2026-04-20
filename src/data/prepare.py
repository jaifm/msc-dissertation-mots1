"""Data preparation: clean raw batches, expand formulation components,
and create reproducible train/test splits.

OVERVIEW
--------
The raw experimental data arrives as two CSV batches (Batch_1 and Batch_2)
collected at different times.  Before any modelling can begin, the data
must be:

  1. **Combined and cleaned**: column names are normalised (spaces →
     underscores, percent signs → 'pct', parentheses removed).  Ghost rows
     (rows with no formulation label) and unnamed trailing columns (pandas
     artefacts from Excel exports) are dropped.

  2. **Formulation expanded**: the raw data encodes the active material
     blend as a categorical label (F1, F2, F3).  These labels are replaced
     by the actual component percentages (NMC_pct, C65_pct, KS6L_pct,
     PVDF_pct) from the hard-coded mapping, which is identical to the
     mapping in the original v1 project.

  3. **Group-aware split**: the slurry dataset contains replicate
     measurements: the same formulation is measured multiple times under
     (nominally) identical conditions.  A naive random split would put
     replicates of the same mix in both train and test, making the
     evaluation optimistic (the model sees nearly-identical rows in
     training).  The group-aware split assigns all replicates of a given
     ``Composite_Mix_ID`` to either train or test, never both.

FORMULATION MAPPING
-------------------
The three base formulations define the NMC / carbon / binder composition:

  F1: NMC 96 % / C65 2 % / KS6L 0 % / PVDF 2 %  (standard ratio)
  F2: NMC 97 % / C65 1.5 % / KS6L 0 % / PVDF 1.5 %  (leaner binder)
  F3: NMC 96 % / C65 0 % / KS6L 2 % / PVDF 2 %  (graphite instead of carbon black)

The remaining composition variables (Solid_Content_pct, Solid_Additive_pct,
Dispersent_Type) vary independently within each formulation and are read
directly from the raw CSV.

SPLIT STRATEGY
--------------
Groups are formed from ``Composite_Mix_ID``, which is constructed as:
  {Formulation}_{Dispersent_Type}_{Solid_Content_pct}_{Solid_Additive_pct}

All rows sharing the same Composite_Mix_ID represent replicates of one
unique formulation/processing combination and are kept together in one split.
An 80/20 group-level train/test split is used with a fixed random seed for
reproducibility.

This module replaces the side-effect-heavy script in
``msc-dissertation-project-MOTS1/src/data_preparation.py``.  All logic lives
in pure functions; the CLI entry point at the bottom runs only when invoked
directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Formulation component lookup
# ---------------------------------------------------------------------------

# Maps categorical formulation labels to their numeric dry-weight percentages.
# Keys must match the values in the raw 'Formulation' column after quote
# stripping.  Values must sum to 100 within rounding.
FORMULATION_COMPONENTS: Dict[str, Dict[str, float]] = {
    "F1": {"NMC_pct": 96, "C65_pct": 2, "KS6L_pct": 0, "PVDF_pct": 2},
    "F2": {"NMC_pct": 97, "C65_pct": 1.5, "KS6L_pct": 0, "PVDF_pct": 1.5},
    "F3": {"NMC_pct": 96, "C65_pct": 0, "KS6L_pct": 2, "PVDF_pct": 2},
}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to a consistent snake_case-like format.

    Transformations applied (in order):
      - Strip leading/trailing whitespace
      - Replace spaces with underscores
      - Remove parentheses (Excel sometimes adds units in parentheses)
      - Remove apostrophes (seen in some raw exports)
      - Replace the percent sign with 'pct' (avoids special-character issues
        when column names are used as Python identifiers or YAML keys)
    """
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("'", "")
        .str.replace("%", "pct")
    )
    return df


def build_combined(batch1_path: Path, batch2_path: Path) -> pd.DataFrame:
    """Read both raw batch CSVs, tag them with a source label, and combine.

    The combined DataFrame retains all rows that have a non-null Formulation
    value.  Rows without a formulation label are measurement artefacts (e.g.
    blank separator rows in the original Excel file) and are dropped.

    A ``Source_Batch`` column is added before merging so downstream analysis
    can identify which batch each row came from, which is useful for detecting
    batch effects.

    The ``Composite_Mix_ID`` column is constructed here because it depends on
    the cleaned column names.  It serves as the grouping key for the
    train/test split.

    Parameters
    ----------
    batch1_path, batch2_path:
        Paths to the two raw batch CSV files.

    Returns
    -------
    pd.DataFrame
        Cleaned, combined DataFrame with ``Composite_Mix_ID`` added.
    """
    a = pd.read_csv(batch1_path)
    b = pd.read_csv(batch2_path)
    a["Source_Batch"] = "Batch_1"
    b["Source_Batch"] = "Batch_2"

    df = pd.concat([a, b], ignore_index=True)
    # Drop rows where Formulation is NaN — these are blank separator rows.
    df = df.dropna(subset=["Formulation"]).copy()
    # Drop columns whose names start with 'Unnamed' — pandas assigns these to
    # empty trailing columns from Excel exports.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = _clean_columns(df)

    # Remove stray apostrophes from categorical text columns that sometimes
    # appear in copy-pasted Excel data (e.g. 'Hypermer → Hypermer).
    for col in ("Formulation", "Dispersent_Type"):
        if col in df.columns:
            df[col] = df[col].str.replace("'", "")

    # Composite_Mix_ID uniquely identifies a formulation–dispersant–loading
    # combination.  All rows sharing this ID are replicates of the same mix
    # and will be assigned to the same train/test split.
    df["Composite_Mix_ID"] = (
        df["Formulation"].astype(str)
        + "_" + df["Dispersent_Type"].astype(str)
        + "_" + df["Solid_Content_pct"].astype(str)
        + "_" + df["Solid_Additive_pct"].astype(str)
    )
    return df


def expand_formulation(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the categorical Formulation column with numeric component columns.

    The Formulation label (F1/F2/F3) is a compact encoding of four percentage
    columns.  Expanding it into explicit numeric columns makes the composition
    directly available as a feature to the machine-learning models and removes
    the need for any special handling of the Formulation column downstream.

    The original Formulation column is dropped after expansion because the
    numeric columns carry all the same information in a more usable form.

    Parameters
    ----------
    df:
        DataFrame containing a 'Formulation' column with values in
        FORMULATION_COMPONENTS.  Rows with unrecognised formulation labels
        will get NaN for all component columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with NMC_pct, C65_pct, KS6L_pct, PVDF_pct added and
        Formulation dropped.
    """
    out = df.copy()
    for comp in ("NMC_pct", "C65_pct", "KS6L_pct", "PVDF_pct"):
        out[comp] = out["Formulation"].map(
            lambda f: FORMULATION_COMPONENTS.get(f, {}).get(comp)
        )
    return out.drop(columns=["Formulation"])


def group_split(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    group_column: str = "Composite_Mix_ID",
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Split on unique groups so that replicates never cross the train/test boundary.

    A random row-level split would be incorrect here because multiple rows
    share the same Composite_Mix_ID (replicate measurements of the same mix).
    Splitting at the row level would put replicates of the same mix in both
    train and test, making the test set non-independent and inflating R².

    This function splits the set of unique group IDs into 80% train / 20%
    test, then assigns every row belonging to a training group to the train
    split and every row belonging to a test group to the test split.

    An overlap check is included to catch any edge case where the split logic
    accidentally places a group in both sets.

    Parameters
    ----------
    df:
        Combined expanded DataFrame with a group column.
    out_dir:
        Directory in which to write ``expanded_train.csv``,
        ``expanded_test.csv``, ``expanded_split_manifest.csv``, and
        ``expanded_split_metadata.json``.
    group_column:
        Column name containing the group identifier.
    test_size:
        Fraction of groups to assign to the test split.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Metadata dict with train/test row and group counts, seed, and test_size.
    """
    groups = df[group_column].dropna().unique()
    train_g, test_g = train_test_split(groups, test_size=test_size, random_state=seed)
    train_set, test_g = set(train_g), set(test_g)

    # Paranoia check: if any group appears in both splits, the downstream
    # evaluation will be silently invalid.
    overlap = train_set & test_g
    if overlap:
        raise RuntimeError("Group overlap after split — seed issue?")

    train_df = df[df[group_column].isin(train_set)].copy()
    test_df = df[df[group_column].isin(test_g)].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "expanded_train.csv"
    test_path = out_dir / "expanded_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Human-readable manifest listing each group and its assigned split.
    manifest = pd.DataFrame(
        {
            group_column: sorted(train_set) + sorted(test_g),
            "split": ["train"] * len(train_set) + ["test"] * len(test_g),
        }
    )
    manifest.to_csv(out_dir / "expanded_split_manifest.csv", index=False)

    meta = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_groups": len(train_set),
        "test_groups": len(test_g),
        "seed": seed,
        "test_size": test_size,
    }
    (out_dir / "expanded_split_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(raw_dir: Path, out_dir: Path) -> None:
    """End-to-end pipeline: combine raw batches → expand → split → save."""
    combined = build_combined(raw_dir / "batch1_raw.csv", raw_dir / "batch2_raw.csv")
    expanded = expand_formulation(combined)
    expanded.to_csv(out_dir / "combined_slurry_data_expanded.csv", index=False)
    meta = group_split(expanded, out_dir / "splits")
    print("Split metadata:", meta)


if __name__ == "__main__":
    import argparse
    from ..paths import PROJECT_ROOT

    # Defaults point at data/ inside this project so the script works without
    # any arguments as long as raw CSVs are present at data/raw/.
    p = argparse.ArgumentParser(
        description="Prepare raw batch CSVs into train/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing batch1_raw.csv and batch2_raw.csv.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for processed files and splits subdirectory.",
    )
    args = p.parse_args()
    main(args.raw, args.out)
