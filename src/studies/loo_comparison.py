"""Leave-one-Composite_Mix_ID-out comparison: augmented vs non-augmented.

For each LOO split in ``splits_chemistry/loo/`` (one per chemistry), runs:

  - RAW: no augmentation (n_aug_points=0), log1p targets
  - AUG: ``n_aug`` augmented points, all configured oracles, log1p targets

with the learner selected by ``params['learner']`` ('xgb' or 'gpr' — the
same study covers both students so a direct k=0 head-to-head is possible).

Results are NOT persisted to the experiment manifest (persist=False) to
avoid creating one run directory per fold; a single summary CSV is written
instead.

GPR note: GPR training is substantially slower than XGBoost (~10-30 s per
fold vs ~1 s).  Expect 1-3 hours for the full 67-fold GPR pass.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    # Student model: 'xgb' or 'gpr'.
    "learner": "xgb",
    # Experiment configs providing the raw / augmented settings.
    "raw_experiment": "exp_033_pa_no_augmentation",
    "aug_experiment": "exp_030_pa_log_target",
    "n_aug": 30,
    # Directory of LOO splits (one subdirectory per held-out chemistry).
    "loo_dir": "data/processed/splits_chemistry/loo",
    "out_csv": "data/processed/xgboost_experiments/loo_comparison_results.csv",
    # When the learner is 'gpr', a head-to-head summary is printed against
    # this CSV (the XGBoost pass) if it exists.
    "compare_csv": None,
    # Run only the first N folds (None = all).  Useful for smoke tests.
    "limit": None,
}


def _make_cfg(base_exp: str, mix_id: str, condition: str, params: Dict[str, Any]):
    from ..config import load_config
    cfg = load_config(base_exp)
    cfg.id = f"loo_{params['learner']}_{condition}_{mix_id}"
    cfg.paths.splits = str(Path(params["loo_dir"]) / mix_id)
    cfg.paths.train_csv = "train.csv"
    cfg.paths.test_csv = "test.csv"
    cfg.physics_augmented.learners = [params["learner"]]
    if condition == "aug":
        cfg.physics_augmented.n_aug_points = int(params["n_aug"])
    return cfg


def _run_fold(mix_id: str, params: Dict[str, Any]) -> List[dict]:
    """Run raw + augmented for one LOO fold.  Returns list of result dicts."""
    from ..experiments.runner import run_experiment

    learner_suffix = f"__{params['learner']}"
    rows: List[dict] = []
    for condition, base_exp in [
        ("raw", params["raw_experiment"]),
        ("aug", params["aug_experiment"]),
    ]:
        cfg = _make_cfg(base_exp, mix_id, condition, params)
        try:
            results = run_experiment(cfg, persist=False)
            ms = results.sections["physics_augmented"].model_summary
            sort_col = "test_r2_log" if "test_r2_log" in ms.columns else "test_r2"
            best = ms.sort_values(sort_col, ascending=False).iloc[0]

            for _, row in ms.iterrows():
                rows.append({
                    "mix_id":    mix_id,
                    "learner":   params["learner"],
                    "condition": condition,
                    "oracle":    row["variant"].replace(learner_suffix, ""),
                    "is_best":   row["variant"] == best["variant"],
                    "r2_raw":    float(row["test_r2"]),
                    "r2_log":    float(row["test_r2_log"]) if "test_r2_log" in row.index else float("nan"),
                    "rmse":      float(row["test_rmse"]),
                    "mae":       float(row["test_mae"]),
                    "n_test":    int(row["n_test"]),
                })
        except Exception:
            print(f"    ERROR [{condition}] {mix_id}:")
            traceback.print_exc()
            rows.append({
                "mix_id": mix_id, "learner": params["learner"],
                "condition": condition, "oracle": "ERROR", "is_best": False,
                "r2_raw": float("nan"), "r2_log": float("nan"),
                "rmse": float("nan"), "mae": float("nan"), "n_test": 0,
            })
    return rows


def run(params: Dict[str, Any]) -> pd.DataFrame:
    loo_dir = PROJECT_ROOT / params["loo_dir"]
    out_csv = PROJECT_ROOT / params["out_csv"]

    mix_ids = sorted(p.name for p in loo_dir.iterdir() if p.is_dir())
    limit: Optional[int] = params.get("limit")
    if limit:
        mix_ids = mix_ids[: int(limit)]
    n = len(mix_ids)
    print(f"LOO comparison ({params['learner']}): {n} folds × 2 conditions (raw / aug)")

    all_rows: List[dict] = []
    t_start = time.perf_counter()

    for i, mix_id in enumerate(mix_ids, 1):
        t0 = time.perf_counter()
        rows = _run_fold(mix_id, params)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)

        raw_r2 = next((r["r2_log"] for r in rows if r["condition"] == "raw" and r["is_best"]), float("nan"))
        aug_r2 = next((r["r2_log"] for r in rows if r["condition"] == "aug" and r["is_best"]), float("nan"))
        aug_ora = next((r["oracle"] for r in rows if r["condition"] == "aug" and r["is_best"]), "?")
        n_test = next((r["n_test"] for r in rows if r["condition"] == "raw"), 0)
        delta = aug_r2 - raw_r2 if (aug_r2 == aug_r2 and raw_r2 == raw_r2) else float("nan")
        print(
            f"[{i:2d}/{n}] {mix_id:<38s}  n={n_test:2d}  "
            f"raw={raw_r2:.3f}  aug={aug_r2:.3f}  Δ={delta:+.3f}  "
            f"({aug_ora})  {elapsed:.1f}s"
        )

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    total = time.perf_counter() - t_start
    print(f"\n{'='*72}")
    print(f"AGGREGATE SUMMARY (best oracle per fold, learner={params['learner']})")
    print(f"{'='*72}")

    best_rows = df[df["is_best"]].copy()
    for cond in ["raw", "aug"]:
        sub = best_rows[best_rows["condition"] == cond]["r2_log"].dropna()
        print(f"\n  {cond.upper():4s}  mean={sub.mean():.4f}  std={sub.std():.4f}  "
              f"median={sub.median():.4f}  min={sub.min():.4f}  max={sub.max():.4f}  n={len(sub)}")

    merged = (
        best_rows[best_rows["condition"] == "aug"][["mix_id", "r2_log"]]
        .rename(columns={"r2_log": "aug"})
        .merge(
            best_rows[best_rows["condition"] == "raw"][["mix_id", "r2_log"]]
            .rename(columns={"r2_log": "raw"}),
            on="mix_id",
        )
    )
    if not merged.empty:
        merged["delta"] = merged["aug"] - merged["raw"]
        print(f"\n  Mean Δ R²(log) [aug − raw]: {merged['delta'].mean():+.4f}")
        print(f"  Folds where aug > raw: {(merged['delta'] > 0).sum()}/{len(merged)}")
        print(f"  Folds where raw > aug: {(merged['delta'] < 0).sum()}/{len(merged)}")

    print("\n  Best oracle distribution (aug):")
    for ora, cnt in best_rows[best_rows["condition"] == "aug"]["oracle"].value_counts().items():
        print(f"    {ora:<20s}  {cnt:2d} folds")

    # Optional head-to-head against a previous pass with a different learner.
    compare_csv = params.get("compare_csv")
    if compare_csv:
        other_path = PROJECT_ROOT / compare_csv
        if other_path.exists():
            other = pd.read_csv(other_path)
            other_best = other[other["is_best"]]
            print(f"\n{'='*72}")
            print(f"HEAD-TO-HEAD vs {other_path.name} (median R²(log) across folds)")
            print(f"{'='*72}")
            for cond in ["raw", "aug"]:
                this_sub = best_rows[best_rows["condition"] == cond]["r2_log"].dropna()
                other_sub = other_best[other_best["condition"] == cond]["r2_log"].dropna()
                print(
                    f"  {cond:<5} this={this_sub.median():.3f}  "
                    f"other={other_sub.median():.3f}  "
                    f"Δ={this_sub.median() - other_sub.median():+.3f}"
                )

    mins, secs = divmod(int(total), 60)
    print(f"\n  Total time: {mins}m {secs:02d}s")
    print(f"  Results saved → {out_csv}")
    return df
