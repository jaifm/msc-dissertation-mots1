"""GPR uncertainty quantification and active-learning proxy analysis.

Three analyses on the parameter-prediction GPR's chemistry-split test
predictions (exp_037 by default):

1. Calibration: empirical vs nominal coverage at 68% and 95% confidence
   intervals, broken down per physics model and per target parameter.

2. σ-error correlation: Spearman ρ between per-recipe GPR predictive
   uncertainty σ(target) and per-recipe absolute prediction error |y−ŷ|,
   aggregated across shear rates (parameters are recipe-level constants).
   Focus target: n_flow (power-law exponent), the bottleneck parameter.

3. Active-learning proxy: rank the held-out test recipes by σ(n_flow) and
   compare the cumulative MAE of σ-guided selection against the random-
   selection baseline — validating σ as an acquisition signal for future
   experimental campaigns.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    # Experiment whose latest persisted run provides gpr_baseline_predictions.csv.
    "source_experiment": "exp_037_gpr_chemistry_split",
    # Explicit run directory (relative to project root) overriding the
    # latest-run auto-detection.  None = auto-detect.
    "run_dir": None,
    "focus_target": "n_flow",
    "focus_model": "power_law",
    "seed": 42,
    "calibration_csv": "data/processed/xgboost_experiments/gpr_uq_calibration.csv",
    "correlation_csv": "data/processed/xgboost_experiments/gpr_uq_sigma_error_corr.csv",
    "active_learning_csv": "data/processed/xgboost_experiments/gpr_uq_active_learning.csv",
}


def _locate_predictions(params: Dict[str, Any]) -> Path:
    if params.get("run_dir"):
        pred = PROJECT_ROOT / params["run_dir"] / "gpr_baseline_predictions.csv"
        if not pred.exists():
            raise FileNotFoundError(f"No GPR predictions at {pred}")
        return pred
    experiments_root = PROJECT_ROOT / "data" / "processed" / "xgboost_experiments"
    pattern = str(experiments_root / f"{params['source_experiment']}_2*")
    for run_dir in sorted(glob.glob(pattern), reverse=True):
        pred = Path(run_dir) / "gpr_baseline_predictions.csv"
        if pred.exists():
            return pred
    raise FileNotFoundError(
        f"No persisted run of {params['source_experiment']!r} with "
        f"gpr_baseline_predictions.csv found under {experiments_root}"
    )


def _recipe_level(df: pd.DataFrame, model_key: str, target: str) -> pd.DataFrame:
    """Aggregate repeated shear-rate rows to one row per recipe."""
    sub = df[(df["model_key"] == model_key) & (df["target"] == target)].copy()
    rec = sub.groupby("mix_id").agg(
        y_true=("y_true", "mean"),
        y_pred=("y_pred", "first"),
        y_std=("y_std", "first"),
        y_lower_68=("y_lower_68", "first"),
        y_upper_68=("y_upper_68", "first"),
        y_lower_95=("y_lower_95", "first"),
        y_upper_95=("y_upper_95", "first"),
    ).reset_index()
    rec["abs_error"] = (rec["y_true"] - rec["y_pred"]).abs()
    return rec


def calibration_check(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical coverage per model × target."""
    rows = []
    for model_key in df["model_key"].unique():
        for target in df["target"].unique():
            sub = df[(df["model_key"] == model_key) & (df["target"] == target)]
            if sub.empty:
                continue
            cov68 = ((sub["y_true"] >= sub["y_lower_68"]) &
                     (sub["y_true"] <= sub["y_upper_68"])).mean()
            cov95 = ((sub["y_true"] >= sub["y_lower_95"]) &
                     (sub["y_true"] <= sub["y_upper_95"])).mean()
            rows.append({
                "model_key": model_key,
                "target": target,
                "n_obs": len(sub),
                "empirical_cov68": round(float(cov68), 4),
                "empirical_cov95": round(float(cov95), 4),
                "over_covered_68": cov68 > 0.68,
                "over_covered_95": cov95 > 0.95,
            })
    return pd.DataFrame(rows)


def sigma_error_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ between recipe-level σ and |y_true − y_pred|."""
    from scipy import stats
    rows = []
    for model_key in df["model_key"].unique():
        for target in df["target"].unique():
            rec = _recipe_level(df, model_key, target)
            if len(rec) < 5:
                continue
            rho, pval = stats.spearmanr(rec["y_std"], rec["abs_error"])
            rows.append({
                "model_key": model_key,
                "target": target,
                "n_recipes": len(rec),
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 4),
                "significant": pval < 0.05,
            })
    return pd.DataFrame(rows)


def active_learning_proxy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Rank test recipes by σ(focus target) and compare to random order."""
    rec = _recipe_level(df, params["focus_model"], params["focus_target"])
    rec = rec.sort_values("y_std", ascending=False).reset_index(drop=True)

    rec["rank_by_sigma"] = rec.index + 1
    rec["cumulative_mae_sigma"] = rec["abs_error"].expanding().mean()

    # Random baseline: expected cumulative MAE over many random permutations.
    rng = np.random.default_rng(int(params["seed"]))
    n = len(rec)
    random_cummaes = []
    for _ in range(1000):
        perm = rng.permutation(n)
        cuml = rec["abs_error"].to_numpy()[perm]
        random_cummaes.append(np.cumsum(cuml) / np.arange(1, n + 1))
    rec["cumulative_mae_random"] = np.stack(random_cummaes).mean(axis=0)
    rec["gain_vs_random"] = rec["cumulative_mae_random"] - rec["cumulative_mae_sigma"]

    return rec[[
        "mix_id", "y_true", "y_pred", "y_std", "abs_error",
        "rank_by_sigma", "cumulative_mae_sigma", "cumulative_mae_random",
        "gain_vs_random",
    ]]


def run(params: Dict[str, Any]) -> pd.DataFrame:
    pred_path = _locate_predictions(params)
    print(f"Loading GPR predictions from {pred_path} …")
    df = pd.read_csv(pred_path)
    print(f"  {len(df)} rows, {df['mix_id'].nunique()} test recipes, "
          f"{df['model_key'].nunique()} models, {df['target'].nunique()} targets")

    print("\n── Calibration (empirical vs nominal coverage) ──")
    calib = calibration_check(df)
    by_model = calib.groupby("model_key")[["empirical_cov68", "empirical_cov95"]].mean()
    print(by_model.round(3).to_string())
    print(f"\n  Overall mean  68% coverage: {calib['empirical_cov68'].mean():.3f}  (nominal 0.680)")
    print(f"  Overall mean  95% coverage: {calib['empirical_cov95'].mean():.3f}  (nominal 0.950)")

    print("\n── Spearman ρ (σ vs |error|) ──")
    corr = sigma_error_correlation(df)
    focus = corr[(corr["model_key"] == params["focus_model"])
                 & (corr["target"] == params["focus_target"])]
    if not focus.empty:
        row = focus.iloc[0]
        print(f"  {params['focus_model']} / {params['focus_target']}: "
              f"ρ={row['spearman_rho']:.3f}, p={row['p_value']:.3f}, n={row['n_recipes']}")

    print("\n── Active-learning proxy (σ-ranked recipe selection) ──")
    al = active_learning_proxy(df, params)
    print(al[["mix_id", "y_std", "abs_error", "cumulative_mae_sigma",
              "cumulative_mae_random", "gain_vs_random"]].round(4).to_string(index=False))
    early_gain = al["gain_vs_random"].iloc[len(al) // 4]
    print(f"\n  Early gain (first 25%): {early_gain:.4f}")
    print(f"  Final gain (all selected): {al['gain_vs_random'].iloc[-1]:.4f} (≈ 0 at saturation)")

    for frame, key in [(calib, "calibration_csv"),
                       (corr, "correlation_csv"),
                       (al, "active_learning_csv")]:
        out = PROJECT_ROOT / params[key]
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        print(f"Saved → {out}")
    return calib
