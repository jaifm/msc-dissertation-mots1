"""GPR baseline section runner.

Mirrors ``sweep.run_baseline`` but drives ``gpr.fit_and_evaluate_gpr`` instead
of ``tuning.tune_and_evaluate_xgb``.  The output is a ``SectionResult`` named
``'gpr_baseline'`` with the same three DataFrames (model_summary, per_target,
predictions) so it can be persisted and compared by the same notebook cells
that display XGBoost results.

The ``y_std`` column in ``predictions`` (GPR posterior standard deviation) is
an extra field that the XGBoost predictions do not carry — downstream plots
can use it to visualise uncertainty, but it is silently ignored by code that
only uses ``y_true`` / ``y_pred``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..curve_metrics import compute_model_curve_rmse
from ..gpr import fit_and_evaluate_gpr
from .results import SectionResult


def run_gpr_baseline(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]],
    cfg: Config,
) -> SectionResult:
    """Run one GPR per (physics model, target) and collect results.

    Parameters
    ----------
    physics_datasets:
        Nested dict ``{model_key: {'train': df, 'test': df}}`` produced by
        ``build_all_physics_datasets``.
    cfg:
        Experiment configuration.  ``cfg.gpr`` controls kernel and restarts.

    Returns
    -------
    SectionResult named ``'gpr_baseline'``.
    """
    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_preds: List[pd.DataFrame] = []
    curve_summary_rows: List[dict] = []

    print(
        f"\n{'='*70}\n"
        f"GPR Baseline  (kernel={cfg.gpr.kernel}, "
        f"n_restarts={cfg.gpr.n_restarts}, "
        f"normalize_y={cfg.gpr.normalize_y})\n"
        f"{'='*70}"
    )

    for model_key in cfg.physics.models:
        display_name = cfg.display_name(model_key)
        train_physics = physics_datasets[model_key]["train"]
        test_physics = physics_datasets[model_key]["test"]

        model_rows: List[dict] = []
        model_pred_list: List[pd.DataFrame] = []
        for target_col in train_physics.attrs["target_cols"]:
            result, preds, _ = fit_and_evaluate_gpr(
                train_physics, test_physics, target_col, cfg,
            )
            result["model_name"] = display_name
            result["model_key"] = model_key
            result["variant"] = "gpr_baseline"
            per_target_rows.append(result)
            model_rows.append(result)

            preds = preds.assign(
                model_name=display_name,
                model_key=model_key,
                variant="gpr_baseline",
            )
            all_preds.append(preds)
            model_pred_list.append(preds)

            print(
                f"  {display_name:18s} | {target_col:20s} "
                f"| R²_train={result['train_r2']:.3f}  R²_test={result['test_r2']:.3f}"
            )

        if model_rows:
            model_summary_rows.append({
                "variant": "gpr_baseline",
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

        if model_pred_list:
            try:
                cm = compute_model_curve_rmse(
                    model_key,
                    pd.concat(model_pred_list, ignore_index=True),
                    test_physics,
                    cfg,
                )
                row = {
                    "variant": "gpr_baseline",
                    "model_name": display_name,
                    "model_key": model_key,
                    "mean_curve_rmse_cross": cm["mean_curve_rmse_cross"],
                    "mean_curve_rmse_within": cm["mean_curve_rmse_within"],
                    "visc_r2": cm["visc_r2"],
                }
                for sr, r2 in zip(cm["visc_r2_shear_rates"], cm["visc_r2_per_shear"]):
                    row[f"visc_r2_at_{int(sr)}"] = r2
                curve_summary_rows.append(row)
            except Exception as exc:
                print(f"  [curve RMSE] {display_name}: {exc}")

    return SectionResult(
        name="gpr_baseline",
        model_summary=(
            pd.DataFrame(model_summary_rows)
            .sort_values("test_r2", ascending=False)
            .reset_index(drop=True)
        ),
        per_target=pd.DataFrame(per_target_rows),
        predictions=(
            pd.concat(all_preds, ignore_index=True) if all_preds else None
        ),
        curve_summary=(
            pd.DataFrame(curve_summary_rows) if curve_summary_rows else None
        ),
    )
