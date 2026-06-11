"""Cross-batch generalisation: Batch_1 → Batch_2 and Batch_2 → Batch_1.

SPLIT DESIGN
------------
Batch_1 and Batch_2 are almost entirely disjoint sets of recipes:
18 recipes appear only in Batch_1, 37 only in Batch_2, and 12 in both.
To keep Composite_Mix_ID leakage-free, the shared recipes go into training
only and are excluded from the test set in both directions:

  batch_b1_to_b2
      train : all Batch_1 rows  (18 B1-only + 12 shared = 30 unique recipes)
      test  : all Batch_2-only rows  (37 unseen recipes)

  batch_b2_to_b1
      train : all Batch_2 rows  (37 B2-only + 12 shared = 49 unique recipes)
      test  : all Batch_1-only rows  (18 unseen recipes)

The split files are generated (and hash-pinned) by
``scripts/prepare_chemistry_splits.py``; this study only reads them.

Each split is run with raw (no augmentation) and aug conditions, then a
combined summary table compares all evaluation regimes (fixed-split, LOO,
batch).
"""
from __future__ import annotations

import glob
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    "raw_experiment": "exp_033_pa_no_augmentation",
    "aug_experiment": "exp_030_pa_log_target",
    "n_aug": 30,
    "splits_root": "data/processed/splits_chemistry",
    "split_names": ["batch_b1_to_b2", "batch_b2_to_b1"],
    "loo_csv": "data/processed/xgboost_experiments/loo_comparison_results.csv",
    "out_csv": "data/processed/xgboost_experiments/batch_comparison_results.csv",
    # Run id of the strict hold-out experiment included in the regime table.
    "strict_holdout_experiment": "exp_035_pa_chemistry_standardised",
}


def _make_cfg(base_exp: str, split_name: str, params: Dict[str, Any]):
    from ..config import load_config
    cfg = load_config(base_exp)
    cfg.id = f"{split_name}_{base_exp}"
    cfg.paths.splits = str(Path(params["splits_root"]) / split_name)
    cfg.paths.train_csv = "train.csv"
    cfg.paths.test_csv = "test.csv"
    return cfg


def _run_one(split_name: str, condition: str, params: Dict[str, Any]) -> dict:
    from ..experiments.runner import run_experiment
    base = params["raw_experiment"] if condition == "raw" else params["aug_experiment"]
    cfg = _make_cfg(base, split_name, params)
    if condition == "aug":
        cfg.physics_augmented.n_aug_points = int(params["n_aug"])

    try:
        results = run_experiment(cfg, persist=False)
        ms = results.sections["physics_augmented"].model_summary
        sort_col = "test_r2_log" if "test_r2_log" in ms.columns else "test_r2"
        best = ms.sort_values(sort_col, ascending=False).iloc[0]
        return {
            "split":     split_name,
            "condition": condition,
            "oracle":    best["variant"].replace("__xgb", ""),
            "r2_raw":    float(best["test_r2"]),
            "r2_log":    float(best[sort_col]),
            "rmse":      float(best["test_rmse"]),
            "mae":       float(best["test_mae"]),
            "n_test":    int(best["n_test"]),
        }
    except Exception:
        traceback.print_exc()
        return {"split": split_name, "condition": condition, "oracle": "ERROR",
                "r2_raw": np.nan, "r2_log": np.nan, "rmse": np.nan,
                "mae": np.nan, "n_test": 0}


def _latest_best(exp_id: str, experiments_root: Path) -> dict | None:
    """Best-oracle row from the most recent persisted run of ``exp_id``."""
    for run_dir in sorted(glob.glob(str(experiments_root / f"{exp_id}_2*")), reverse=True):
        ms_path = Path(run_dir) / "physics_augmented_model_summary.csv"
        if not ms_path.exists():
            continue
        ms = pd.read_csv(ms_path)
        if "test_r2_log" not in ms.columns:
            continue
        best = ms.sort_values("test_r2_log", ascending=False).iloc[0]
        return {
            "r2_log": float(best["test_r2_log"]),
            "r2_raw": float(best["test_r2"]),
            "rmse":   float(best["test_rmse"]),
            "note":   f"n_test={best['n_test']}",
        }
    return None


def _compile_summary(batch_rows: List[dict], params: Dict[str, Any]) -> None:
    experiments_root = PROJECT_ROOT / "data" / "processed" / "xgboost_experiments"
    loo_csv = PROJECT_ROOT / params["loo_csv"]

    print(f"\n{'='*72}")
    print("FULL EVALUATION REGIME COMPARISON")
    print(f"{'='*72}\n")

    rows: List[dict] = []

    for label, exp_id, cond in [
        ("Fixed split", params["raw_experiment"], "raw"),
        ("Fixed split", params["aug_experiment"], "aug"),
    ]:
        best = _latest_best(exp_id, experiments_root)
        if best:
            rows.append({"regime": label, "condition": cond, **best})

    best = _latest_best(params["strict_holdout_experiment"], experiments_root)
    if best:
        rows.append({"regime": "Strict hold-out", "condition": "aug", **best})

    if loo_csv.exists():
        loo = pd.read_csv(loo_csv)
        loo_best = loo[loo["is_best"]]
        for cond in ["raw", "aug"]:
            sub = loo_best[loo_best["condition"] == cond]
            r2 = sub["r2_log"].dropna()
            rows.append({"regime": "LOO (67 folds)", "condition": cond,
                         "r2_log": r2.median(),
                         "r2_raw": sub["r2_raw"].median(),
                         "rmse":   sub["rmse"].median(),
                         "note":   f"median across {len(r2)} folds"})

    for row in batch_rows:
        label = ("Batch_1→2 (37 unseen)" if row["split"] == "batch_b1_to_b2"
                 else "Batch_2→1 (18 unseen)")
        rows.append({"regime": label, "condition": row["condition"],
                     "r2_log": row["r2_log"], "r2_raw": row["r2_raw"],
                     "rmse": row["rmse"], "note": f"n_test={row['n_test']}"})

    summary = pd.DataFrame(rows)
    print(f"{'Regime':<30} {'Cond':<5} {'R²(log)':>8}  {'R²(raw)':>8}  {'RMSE':>7}  Note")
    print("-" * 80)
    for _, r in summary.iterrows():
        print(f"{r['regime']:<30} {r['condition']:<5} {r['r2_log']:>8.3f}  "
              f"{r['r2_raw']:>8.3f}  {r['rmse']:>7.2f}  {r['note']}")


def run(params: Dict[str, Any]) -> pd.DataFrame:
    print("=" * 72)
    print("Cross-batch generalisation study")
    print("=" * 72)

    splits_root = PROJECT_ROOT / params["splits_root"]
    for split_name in params["split_names"]:
        split_dir = splits_root / split_name
        if not (split_dir / "train.csv").exists():
            raise FileNotFoundError(
                f"Batch split missing: {split_dir}. "
                f"Run scripts/prepare_chemistry_splits.py to generate it."
            )

    batch_rows: List[dict] = []
    for split_name in params["split_names"]:
        for condition in ["raw", "aug"]:
            print(f"\n--- {split_name} [{condition}] ---")
            t0 = time.perf_counter()
            row = _run_one(split_name, condition, params)
            elapsed = time.perf_counter() - t0
            batch_rows.append(row)
            print(f"  best={row['oracle']}  R²(log)={row['r2_log']:.3f}  "
                  f"R²(raw)={row['r2_raw']:.3f}  RMSE={row['rmse']:.3f}  ({elapsed:.1f}s)")

    df = pd.DataFrame(batch_rows)
    out_csv = PROJECT_ROOT / params["out_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    _compile_summary(batch_rows, params)
    print(f"\nBatch results saved → {out_csv}")
    return df
