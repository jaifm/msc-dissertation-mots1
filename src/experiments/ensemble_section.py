"""GPR + XGBoost ensemble section runner.

Runs both a GPR and an XGBoost model for each (physics model, target) and
averages their test-set predictions with equal weight (0.5 / 0.5).

GPR and tree models make structurally different errors: GPR interpolates
smoothly in kernel-distance space while XGBoost builds step-function
boundaries driven by split-point gain.  On a small dataset with a mix of
smooth composition trends and discrete-grid structure, their residuals are
partially uncorrelated, so blending reduces total variance.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..gpr import fit_and_evaluate_gpr
from ..pipeline import r2_mse_rmse
from ..tuning import tune_and_evaluate_xgb
from .results import SectionResult


def run_gpr_xgb_ensemble(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]],
    cfg: Config,
) -> SectionResult:
    """Run one GPR + one XGBoost per (physics model, target) and ensemble.

    Parameters
    ----------
    physics_datasets:
        Nested dict ``{model_key: {'train': df, 'test': df}}``.
    cfg:
        Experiment configuration.  Both ``cfg.xgb`` and ``cfg.gpr`` are used.

    Returns
    -------
    SectionResult named ``'gpr_xgb_ensemble'``.
    """
    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_preds: List[pd.DataFrame] = []

    print(
        f"\n{'='*70}\n"
        f"GPR + XGBoost Ensemble  "
        f"(kernel={cfg.gpr.kernel}, "
        f"n_restarts={cfg.gpr.n_restarts}, "
        f"xgb_trials={cfg.xgb.n_trials})\n"
        f"{'='*70}"
    )

    for model_key in cfg.physics.models:
        display_name = cfg.display_name(model_key)
        train_physics = physics_datasets[model_key]["train"]
        test_physics = physics_datasets[model_key]["test"]

        model_rows: List[dict] = []
        for target_col in train_physics.attrs["target_cols"]:
            # --- XGBoost --------------------------------------------------------
            xgb_result, xgb_preds, _ = tune_and_evaluate_xgb(
                train_physics, test_physics, target_col, cfg,
            )
            # --- GPR ------------------------------------------------------------
            gpr_result, gpr_preds, _ = fit_and_evaluate_gpr(
                train_physics, test_physics, target_col, cfg,
            )

            # --- Equal-weight average -------------------------------------------
            y_true = gpr_preds["y_true"].to_numpy()
            y_pred_ens = 0.5 * (
                xgb_preds["y_pred"].to_numpy() + gpr_preds["y_pred"].to_numpy()
            )
            test_r2, test_mse, test_rmse = r2_mse_rmse(y_true, y_pred_ens)

            result = {
                "target": target_col,
                "model_name": display_name,
                "model_key": model_key,
                "variant": "gpr_xgb_ensemble",
                "train_r2": float(np.mean([xgb_result["train_r2"], gpr_result["train_r2"]])),
                "train_mse": float(np.mean([xgb_result["train_mse"], gpr_result["train_mse"]])),
                "train_rmse": float(np.mean([xgb_result["train_rmse"], gpr_result["train_rmse"]])),
                "test_r2": test_r2,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                # Individual model metrics kept for per-target diagnostics.
                "xgb_test_r2": xgb_result["test_r2"],
                "gpr_test_r2": gpr_result["test_r2"],
            }
            per_target_rows.append(result)
            model_rows.append(result)

            preds_df = pd.DataFrame({
                "y_true": y_true,
                "y_pred": y_pred_ens,
                "y_pred_xgb": xgb_preds["y_pred"].to_numpy(),
                "y_pred_gpr": gpr_preds["y_pred"].to_numpy(),
                "target": target_col,
                "model_name": display_name,
                "model_key": model_key,
                "variant": "gpr_xgb_ensemble",
            })
            all_preds.append(preds_df)

            print(
                f"  {display_name:18s} | {target_col:20s} "
                f"| XGB={xgb_result['test_r2']:.3f}  "
                f"GPR={gpr_result['test_r2']:.3f}  "
                f"ENS={test_r2:.3f}"
            )

        if model_rows:
            model_summary_rows.append({
                "variant": "gpr_xgb_ensemble",
                "model_name": display_name,
                "model_key": model_key,
                "train_r2": float(np.mean([r["train_r2"] for r in model_rows])),
                "train_mse": float(np.mean([r["train_mse"] for r in model_rows])),
                "train_rmse": float(np.mean([r["train_rmse"] for r in model_rows])),
                "test_r2": float(np.mean([r["test_r2"] for r in model_rows])),
                "test_mse": float(np.mean([r["test_mse"] for r in model_rows])),
                "test_rmse": float(np.mean([r["test_rmse"] for r in model_rows])),
                "n_targets": len(model_rows),
            })

    return SectionResult(
        name="gpr_xgb_ensemble",
        model_summary=(
            pd.DataFrame(model_summary_rows)
            .sort_values("test_r2", ascending=False)
            .reset_index(drop=True)
        ),
        per_target=pd.DataFrame(per_target_rows),
        predictions=(
            pd.concat(all_preds, ignore_index=True) if all_preds else None
        ),
    )
