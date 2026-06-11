"""Few-shot recovery: how much does 1 or 2 measurements of a new recipe help?

For each eligible Composite_Mix_ID, splits exist on disk for k ∈ {1, 2}:
k batches of the held-out recipe are leaked into training and the remaining
batches form the test set.  Each split is run under the same raw / aug
conditions as the LOO study so the recovery curve k=0 → k=1 → k=2 can be
plotted directly (k=0 comes from the pre-computed LOO comparison CSV).
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    "raw_experiment": "exp_033_pa_no_augmentation",
    "aug_experiment": "exp_030_pa_log_target",
    "n_aug": 30,
    "few_shot_dir": "data/processed/splits_chemistry/few_shot",
    # LOO results provide the k=0 baseline for the recovery summary.
    "loo_csv": "data/processed/xgboost_experiments/loo_comparison_results.csv",
    "out_csv": "data/processed/xgboost_experiments/fewshot_comparison_results.csv",
    # Run only the first N splits (None = all).  Useful for smoke tests.
    "limit": None,
}


def _make_cfg(base_exp: str, split_name: str, params: Dict[str, Any]):
    from ..config import load_config
    cfg = load_config(base_exp)
    cfg.id = f"fewshot_{split_name}_{base_exp}"
    cfg.paths.splits = str(Path(params["few_shot_dir"]) / split_name)
    cfg.paths.train_csv = "train.csv"
    cfg.paths.test_csv = "test.csv"
    # Few-shot splits intentionally have the held-out recipe in both train
    # (k leaked rows) and test (remaining rows) — bypass the leakage guard.
    cfg.allow_overlap = True
    return cfg


def _run_fold(split_name: str, params: Dict[str, Any]) -> List[dict]:
    from ..experiments.runner import run_experiment

    # Derive mix_id and k from the folder name, e.g. F1_Hypermer_69.0_0.25_k1.
    k = int(split_name.rsplit("_k", 1)[1])
    mix_id = split_name.rsplit("_k", 1)[0]

    rows: List[dict] = []
    for condition, base_exp in [
        ("raw", params["raw_experiment"]),
        ("aug", params["aug_experiment"]),
    ]:
        cfg = _make_cfg(base_exp, split_name, params)
        if condition == "aug":
            cfg.physics_augmented.n_aug_points = int(params["n_aug"])
        try:
            results = run_experiment(cfg, persist=False)
            ms = results.sections["physics_augmented"].model_summary
            sort_col = "test_r2_log" if "test_r2_log" in ms.columns else "test_r2"
            best = ms.sort_values(sort_col, ascending=False).iloc[0]
            rows.append({
                "mix_id":    mix_id,
                "k":         k,
                "condition": condition,
                "oracle":    best["variant"].replace("__xgb", ""),
                "r2_log":    float(best[sort_col]),
                "r2_raw":    float(best["test_r2"]),
                "rmse":      float(best["test_rmse"]),
                "mae":       float(best["test_mae"]),
                "n_test":    int(best["n_test"]),
            })
        except Exception:
            print(f"    ERROR [{condition}] {split_name}:")
            traceback.print_exc()
            rows.append({
                "mix_id": mix_id, "k": k, "condition": condition,
                "oracle": "ERROR", "r2_log": float("nan"),
                "r2_raw": float("nan"), "rmse": float("nan"),
                "mae": float("nan"), "n_test": 0,
            })
    return rows


def run(params: Dict[str, Any]) -> pd.DataFrame:
    few_shot_dir = PROJECT_ROOT / params["few_shot_dir"]
    out_csv = PROJECT_ROOT / params["out_csv"]
    loo_csv = PROJECT_ROOT / params["loo_csv"]

    split_names = sorted(p.name for p in few_shot_dir.iterdir() if p.is_dir())
    limit: Optional[int] = params.get("limit")
    if limit:
        split_names = split_names[: int(limit)]
    n = len(split_names)
    print(f"Few-shot comparison: {n} splits × 2 conditions (raw / aug)")

    all_rows: List[dict] = []
    t_start = time.perf_counter()

    for i, split_name in enumerate(split_names, 1):
        t0 = time.perf_counter()
        rows = _run_fold(split_name, params)
        elapsed = time.perf_counter() - t0
        all_rows.extend(rows)

        raw_r2 = next((r["r2_log"] for r in rows if r["condition"] == "raw"), float("nan"))
        aug_r2 = next((r["r2_log"] for r in rows if r["condition"] == "aug"), float("nan"))
        k = rows[0]["k"] if rows else "?"
        mix_id = rows[0]["mix_id"] if rows else split_name
        print(
            f"[{i:2d}/{n}] k={k}  {mix_id:<38s}  "
            f"raw={raw_r2:.3f}  aug={aug_r2:.3f}  {elapsed:.1f}s"
        )

    df = pd.DataFrame(all_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*72}")
    print("RECOVERY SUMMARY  (R²(log), median across eligible IDs)")
    print(f"{'='*72}")
    print(f"{'k':<4} {'Condition':<6} {'Median R²(log)':>14}  {'Mean R²(log)':>12}  n")
    print("-" * 50)

    # k=0 baseline from the LOO CSV, restricted to the few-shot-eligible IDs.
    if loo_csv.exists():
        loo = pd.read_csv(loo_csv)
        loo_best = loo[loo["is_best"]]
        eligible_ids = df["mix_id"].unique()
        for cond in ["raw", "aug"]:
            sub = loo_best[
                (loo_best["condition"] == cond)
                & (loo_best["mix_id"].isin(eligible_ids))
            ]["r2_log"].dropna()
            print(f"{'0':<4} {cond:<6} {sub.median():>14.3f}  {sub.mean():>12.3f}  {len(sub)}")

    for k in sorted(df["k"].unique()):
        sub_k = df[df["k"] == k]
        for cond in ["raw", "aug"]:
            sub = sub_k[sub_k["condition"] == cond]["r2_log"].dropna()
            print(f"{k:<4} {cond:<6} {sub.median():>14.3f}  {sub.mean():>12.3f}  {len(sub)}")

    total = time.perf_counter() - t_start
    mins, secs = divmod(int(total), 60)
    print(f"\n  Total time: {mins}m {secs:02d}s")
    print(f"  Results saved → {out_csv}")
    return df
