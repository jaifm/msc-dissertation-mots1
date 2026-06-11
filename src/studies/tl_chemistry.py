"""Transfer learning vs physics augmentation on identical chemistry-split rows.

This study answers the dissertation's RQ2 head-to-head question: under the
standardised chemistry split, does the El-Dalahmeh-style transfer-learning
pipeline match the physics-augmented model on the SAME held-out recipes?

Prior comparisons were confounded: the TL replication (exp_027/028) was
evaluated under the paper's batch-based protocol while the physics-augmented
results used the chemistry split, so their R² values were not directly
comparable.  Here the TL pipeline is transplanted onto the chemistry split:

  * inputs: the paper's four-feature subset (Formulation, Dispersent_Type,
    Solid_Content_pct, Solid_Dispersant_pct),
  * mechanism: LightGBM continuation training — pretrain on the chemistry-
    train rows that came from Batch_1 (500 trees, lr 0.10), fine-tune on the
    chemistry-train rows from Batch_2 (up to 100 trees, lr 0.05), exactly
    the paper's Table-3 recipe,
  * one model per measured shear rate (the TL paradigm is shear-blind:
    γ̇ is not an input, so three separate models are required and the
    pipeline cannot predict between or beyond them),
  * evaluation: the chemistry-split test rows — identical to exp_035.

Variants:
  * ``tl_lightgbm``      — pretrain Batch_1 → fine-tune Batch_2 (paper's TL)
  * ``pooled_lightgbm``  — single-stage LightGBM on all train rows (no TL)
  * ``ridge``            — regularised linear baseline

Each variant runs with the log1p target transform (matching the
physics-augmented headline metric) and raw Pa·s for reference.  The summary
prints pooled R²(log) next to the physics-augmented best-oracle figure read
from the latest persisted run of ``pa_experiment``.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    # Chemistry split evaluated — identical rows to the PA headline runs.
    "split_dir": "data/processed/splits_chemistry/default",
    # Physics-augmented experiment whose latest persisted run provides the
    # comparison row in the summary table.
    "pa_experiment": "exp_035_pa_chemistry_standardised",
    "variants": ["tl_lightgbm", "pooled_lightgbm", "ridge"],
    "target_transforms": ["log1p_nonnegative", "none"],
    # Paper Table 3 hyperparameters.
    "lightgbm": {
        "n_base_trees": 500,
        "n_finetune_trees": 100,
        "learning_rate_base": 0.10,
        "learning_rate_finetune": 0.05,
    },
    "source_batch": "Batch_1",
    "target_batch": "Batch_2",
    "seed": 42,
    "out_csv": "data/processed/xgboost_experiments/tl_chemistry_results.csv",
}

_SHEAR_COLS = {
    1.0:   "Viscosity_at_shear_rate_1_1/s",
    10.0:  "Viscosity_at_shear_rate_10_1/s",
    100.0: "Viscosity_at_shear_rate_100_1/s",
}


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / ss_tot


def _fit_predict(
    variant: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    train_batches: pd.Series,
    params: Dict[str, Any],
) -> np.ndarray:
    from ..transfer_learning.learners import make_learner

    seed = int(params["seed"])
    if variant == "ridge":
        learner = make_learner("ridge")
        learner.fit(X_train.to_numpy(float), y_train)
        return learner.predict(X_test.to_numpy(float))

    lgb_kwargs = dict(params["lightgbm"])
    learner = make_learner("lightgbm", seed=seed, **lgb_kwargs)
    if variant == "tl_lightgbm":
        src_mask = (train_batches == params["source_batch"]).to_numpy()
        tgt_mask = (train_batches == params["target_batch"]).to_numpy()
        learner.fit(
            X_train.to_numpy(float)[src_mask], y_train[src_mask],
            X_train.to_numpy(float)[tgt_mask], y_train[tgt_mask],
        )
    elif variant == "pooled_lightgbm":
        learner.fit(X_train.to_numpy(float), y_train)
    else:
        raise ValueError(f"Unknown variant {variant!r}")
    return learner.predict(X_test.to_numpy(float))


def _pa_comparison_row(params: Dict[str, Any]) -> Optional[dict]:
    experiments_root = PROJECT_ROOT / "data" / "processed" / "xgboost_experiments"
    pattern = str(experiments_root / f"{params['pa_experiment']}_2*")
    for run_dir in sorted(glob.glob(pattern), reverse=True):
        ms_path = Path(run_dir) / "physics_augmented_model_summary.csv"
        if not ms_path.exists():
            continue
        ms = pd.read_csv(ms_path)
        if "test_r2_log" not in ms.columns:
            continue
        best = ms.sort_values("test_r2_log", ascending=False).iloc[0]
        return {
            "variant": f"physics_augmented ({best['variant']})",
            "r2_log_pooled": float(best["test_r2_log"]),
            "r2_raw_pooled": float(best["test_r2"]),
        }
    return None


def run(params: Dict[str, Any]) -> pd.DataFrame:
    from ..transfer_learning.data import (
        PAPER_INPUT_COLS, one_hot_encode_inputs,
    )

    split_dir = PROJECT_ROOT / params["split_dir"]
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    for df in (train_df, test_df):
        if "Solid_Dispersant_pct" not in df.columns:
            df["Solid_Dispersant_pct"] = df["Solid_Additive_pct"]

    overlap = (set(train_df["Composite_Mix_ID"].unique())
               & set(test_df["Composite_Mix_ID"].unique()))
    assert not overlap, f"chemistry leakage: {sorted(overlap)[:3]}"

    print("=" * 72)
    print("TL vs physics augmentation — identical chemistry-split test rows")
    print("=" * 72)
    print(f"Train: {len(train_df)} rows "
          f"({(train_df['Source_Batch'] == params['source_batch']).sum()} "
          f"{params['source_batch']}, "
          f"{(train_df['Source_Batch'] == params['target_batch']).sum()} "
          f"{params['target_batch']})")
    print(f"Test:  {len(test_df)} rows, "
          f"{test_df['Composite_Mix_ID'].nunique()} held-out recipes")

    X_train_enc, cats = one_hot_encode_inputs(train_df, PAPER_INPUT_COLS)
    X_test_enc, _ = one_hot_encode_inputs(test_df, PAPER_INPUT_COLS,
                                          fit_categories=cats)
    train_batches = train_df["Source_Batch"]

    rows: List[dict] = []
    for variant in params["variants"]:
        for transform in params["target_transforms"]:
            pooled_true_log: List[np.ndarray] = []
            pooled_pred_log: List[np.ndarray] = []
            pooled_true_raw: List[np.ndarray] = []
            pooled_pred_raw: List[np.ndarray] = []
            per_shear: Dict[float, Dict[str, float]] = {}

            for shear, col in _SHEAR_COLS.items():
                y_train = train_df[col].to_numpy(float)
                y_test = test_df[col].to_numpy(float)
                if transform == "log1p_nonnegative":
                    y_fit = np.log1p(np.maximum(y_train, 0.0))
                else:
                    y_fit = y_train

                y_pred_fit = _fit_predict(
                    variant, X_train_enc, y_fit, X_test_enc,
                    train_batches, params,
                )
                if transform == "log1p_nonnegative":
                    y_pred = np.maximum(np.expm1(y_pred_fit), 0.0)
                else:
                    y_pred = y_pred_fit

                yt_log = np.log1p(np.maximum(y_test, 0.0))
                yp_log = np.log1p(np.maximum(y_pred, 0.0))
                per_shear[shear] = {
                    "r2_raw": _r2(y_test, y_pred),
                    "r2_log": _r2(yt_log, yp_log),
                }
                pooled_true_raw.append(y_test)
                pooled_pred_raw.append(y_pred)
                pooled_true_log.append(yt_log)
                pooled_pred_log.append(yp_log)

            r2_log_pooled = _r2(np.concatenate(pooled_true_log),
                                np.concatenate(pooled_pred_log))
            r2_raw_pooled = _r2(np.concatenate(pooled_true_raw),
                                np.concatenate(pooled_pred_raw))
            print(f"\n--- {variant} [{transform}] ---")
            for shear in sorted(per_shear):
                m = per_shear[shear]
                print(f"  γ̇={shear:6.1f}  R²(log)={m['r2_log']:.3f}  "
                      f"R²(raw)={m['r2_raw']:.3f}")
            print(f"  POOLED    R²(log)={r2_log_pooled:.3f}  "
                  f"R²(raw)={r2_raw_pooled:.3f}")

            row = {
                "variant": variant,
                "target_transform": transform,
                "r2_log_pooled": round(r2_log_pooled, 4),
                "r2_raw_pooled": round(r2_raw_pooled, 4),
                "n_test_points": int(len(test_df) * len(_SHEAR_COLS)),
            }
            for shear in sorted(per_shear):
                row[f"r2_log_shear_{shear:g}"] = round(per_shear[shear]["r2_log"], 4)
                row[f"r2_raw_shear_{shear:g}"] = round(per_shear[shear]["r2_raw"], 4)
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / params["out_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\n{'='*72}")
    print("HEAD-TO-HEAD (pooled R²(log) on identical test rows)")
    print(f"{'='*72}")
    pa_row = _pa_comparison_row(params)
    if pa_row:
        print(f"  {pa_row['variant']:<45s} {pa_row['r2_log_pooled']:.3f}")
    best_log = df[df["target_transform"] == "log1p_nonnegative"]
    for _, r in best_log.iterrows():
        print(f"  {r['variant'] + ' [log1p]':<45s} {r['r2_log_pooled']:.3f}")
    print("\n  NOTE: the TL pipeline trains one model per measured shear and "
          "cannot be queried\n  at unmeasured shear rates; the comparison "
          "covers only the three measured points.")
    print(f"\nSaved → {out_csv}")
    return df
