"""Baseline model-comparison sweep and the log-numeric-features variant.

OVERVIEW
--------
This module runs the two main supervised-learning experiment sections:

  ``baseline``  -- Trains and evaluates one XGBoost model per physics model
                   (cy5, power_law, sisko, williamson, constrained_cy) per
                   target parameter, using whatever feature settings are
                   specified in the config.  Predictions are collected so
                   the results can be used for residual analysis.

  ``sweep``     -- Runs the same loop twice: once with
                   ``log_numeric_features=False`` (the baseline) and once
                   with ``log_numeric_features=True`` (the log1p variant).
                   The two variants are compared side-by-side to assess
                   whether compressing skewed numeric features improves
                   prediction accuracy.

SHARED LOOP (_run_variants)
----------------------------
Both sections share a single inner loop (``_run_variants``) parametrised by
a list of ``(variant_name, log_numeric_features)`` tuples.  The baseline
section passes a list with one tuple; the sweep section passes two.  This
avoids code duplication while keeping the two public entry points
(``run_baseline``, ``run_sweep``) distinct in the results structure.

OUTPUT STRUCTURE
----------------
Both functions return a ``SectionResult`` with three DataFrames:

  ``model_summary``   -- One row per (model, variant), averaged over all
                         targets.  This is the primary table for ranking
                         models.

  ``per_target``      -- One row per (model, target, variant).  Used for
                         the target-level heatmap and for diagnosing which
                         physics parameters are hardest to predict.

  ``predictions``     -- All test-set (y_true, y_pred) pairs with model and
                         variant labels.  Used for scatter plots and residual
                         analysis.  Only collected for the baseline section;
                         the sweep section omits it to reduce storage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..curve_metrics import compute_model_curve_rmse
from ..tuning import tune_and_evaluate_xgb
from .results import SectionResult


def _run_variants(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]],
    cfg: Config,
    variants: List[Tuple[str, bool]],
    *,
    collect_predictions: bool,
) -> SectionResult:
    """Inner loop shared by both the baseline and sweep sections.

    Iterates over every (variant, model, target) combination, calls
    ``tune_and_evaluate_xgb`` for each, and collects the results into the
    three output DataFrames.

    Parameters
    ----------
    physics_datasets:
        Nested dict ``{model_key: {'train': df, 'test': df}}`` produced by
        ``build_all_physics_datasets``.
    cfg:
        Experiment configuration.
    variants:
        List of ``(variant_name, log_numeric_features)`` tuples.  The
        baseline section passes one tuple; the sweep section passes two.
    collect_predictions:
        Whether to accumulate per-row (y_true, y_pred) pairs.  Set to True
        for the baseline (needed for residual plots) and False for the sweep
        (which runs twice as many models and would double storage).

    Returns
    -------
    SectionResult
        Named 'baseline' or 'sweep' by the calling function.
    """
    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_preds: List[pd.DataFrame] = []
    curve_summary_rows: List[dict] = []

    for variant_name, log_numeric in variants:
        print(f"\n{'='*70}\nVariant: {variant_name} (log_numeric_features={log_numeric})\n{'='*70}")
        for model_key in cfg.physics.models:
            display_name = cfg.display_name(model_key)
            train_physics = physics_datasets[model_key]["train"]
            test_physics = physics_datasets[model_key]["test"]

            model_rows: List[dict] = []
            model_pred_list: List[pd.DataFrame] = []
            for target_col in train_physics.attrs["target_cols"]:
                result, preds, _ = tune_and_evaluate_xgb(
                    train_physics,
                    test_physics,
                    target_col,
                    cfg,
                    log_numeric_features=log_numeric,
                )
                result["model_name"] = display_name
                result["model_key"] = model_key
                result["variant"] = variant_name
                per_target_rows.append(result)
                model_rows.append(result)

                if collect_predictions:
                    preds = preds.assign(
                        model_name=display_name, model_key=model_key, variant=variant_name
                    )
                    all_preds.append(preds)
                    model_pred_list.append(preds)

                print(
                    f"  {display_name:18s} | {target_col:20s} "
                    f"| R²_train={result['train_r2']:.3f} R²_test={result['test_r2']:.3f}"
                )

            if model_rows:
                # Summarise this model by averaging metrics across all its targets.
                # A model with five targets (e.g. CY5) is summarised on equal footing
                # with one that has two targets (Power Law) so comparisons are fair.
                model_summary_rows.append(
                    {
                        "variant": variant_name,
                        "model_name": display_name,
                        "model_key": model_key,
                        "train_r2": float(np.mean([r["train_r2"] for r in model_rows])),
                        "train_mse": float(np.mean([r["train_mse"] for r in model_rows])),
                        "train_rmse": float(np.mean([r["train_rmse"] for r in model_rows])),
                        "test_r2": float(np.mean([r["test_r2"] for r in model_rows])),
                        "test_mse": float(np.mean([r["test_mse"] for r in model_rows])),
                        "test_rmse": float(np.mean([r["test_rmse"] for r in model_rows])),
                        "n_targets": len(model_rows),
                    }
                )

            # Curve RMSE: only when predictions are collected (baseline, not sweep).
            if collect_predictions and model_pred_list:
                try:
                    cm = compute_model_curve_rmse(
                        model_key,
                        pd.concat(model_pred_list, ignore_index=True),
                        test_physics,
                        cfg,
                    )
                    row = {
                        "variant": variant_name,
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
        # Section name is set by the calling public function, not here.
        name="sweep" if len(variants) > 1 else "baseline",
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


def run_baseline(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]], cfg: Config
) -> SectionResult:
    """Run the single-variant baseline sweep.

    Uses ``cfg.features.log_numeric`` as-is (typically False for the
    baseline experiment).  Predictions are collected for all models and
    targets so that residual plots can be generated after the run.

    Parameters
    ----------
    physics_datasets:
        Dict of physics-parameter DataFrames keyed by model identifier.
    cfg:
        Experiment configuration.

    Returns
    -------
    SectionResult named ``'baseline'``.
    """
    # Single-element variant list: only the baseline feature setting is run.
    variants = [("baseline", cfg.features.log_numeric)]
    return _run_variants(physics_datasets, cfg, variants, collect_predictions=True)


def run_sweep(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]], cfg: Config
) -> SectionResult:
    """Run the two-variant log-numeric sweep.

    Trains models twice: once without log1p feature transformation
    (``log_numeric=False``) and once with it (``log_numeric=True``).
    Comparing the two variants reveals whether the right-skewed small-dose
    features (C65_pct, KS6L_pct, Solid_Additive_pct) benefit from
    scale compression.

    Predictions are not collected for the sweep to avoid doubling the
    storage footprint.  Use the baseline section for residual analysis.

    Parameters
    ----------
    physics_datasets:
        Dict of physics-parameter DataFrames keyed by model identifier.
    cfg:
        Experiment configuration.

    Returns
    -------
    SectionResult named ``'sweep'`` with a ``variant`` column distinguishing
    the two runs.
    """
    variants = [
        ("baseline", False),    # standard numeric features
        ("log_numeric", True),  # log1p-transformed numeric features
    ]
    return _run_variants(physics_datasets, cfg, variants, collect_predictions=False)
