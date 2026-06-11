"""Leave-one-out (LOO) generalisability evaluation.

OVERVIEW
--------
The standard train/test split evaluates how well the model generalises to
held-out formulations that were defined before the split.  But in practice,
a researcher wants to know: "if I prepare a brand new formulation, how
accurately can this model predict its rheology before I measure it?"

Leave-one-out evaluation answers that question by treating each unique
formulation (or each individual measurement) as the test set in turn,
training on the rest, and averaging the held-out predictions.  This
stress-tests the model's ability to extrapolate to compositions not seen
during training, which is the relevant use case for slurry design.

TWO LOO VARIANTS
----------------
``leave_out_unit = 'row'``  (LeaveOneOut)
    Each fold holds out exactly one measurement row.  The model is trained
    on all other rows.  This variant has N folds (one per row) and is the
    most data-efficient evaluation, but it may be optimistic because
    replicate measurements of the same mix are split across train and test.
    It answers: "can the model predict a new measurement of a known mix?"

``leave_out_unit = 'mix'``  (LeaveOneGroupOut)
    Each fold holds out all rows sharing the same Composite_Mix_ID (i.e.
    all replicates of one unique formulation).  The model is trained on
    all rows from all other mixes.  This is the stricter and more realistic
    evaluation: it simulates predicting a completely unseen formulation.
    It answers: "can the model predict the rheology of a new formulation?"

``leave_out_unit = 'batch'``  (LeaveOneGroupOut on Source_Batch)
    Two folds only: train on Batch_1, test on Batch_2; then train on
    Batch_2, test on Batch_1.  This is the most demanding evaluation —
    it checks whether the model generalises across entirely different
    experimental campaigns (different dates, potentially different
    instruments or operators).  A high R² here is strong evidence that
    the model has learned true composition–rheology relationships rather
    than batch-specific artefacts.
    It answers: "does the model transfer to a completely new batch?"

FIXED HYPERPARAMETERS
---------------------
LOO runs hundreds of folds (one per row or one per unique mix), so
re-running Optuna tuning inside each fold would be computationally
infeasible.  Instead, a fixed XGBoost parameter set from
``cfg.xgb.fixed_params`` is used.  These parameters should be set to
reasonable defaults in the config (derived from the baseline Optuna results
if possible) to ensure the LOO evaluation is representative.

TARGET SCOPE
------------
``cfg.loo.target_scope = 'first'`` runs LOO only on the first target of
each physics model (e.g. eta_0 for CY5, k_consistency for Power Law).
This reduces runtime significantly for exploratory runs.

``cfg.loo.target_scope = 'all'`` runs LOO on every target of every model,
providing a complete picture at the cost of more computation.

METRICS
-------
LOO metrics are computed by concatenating all held-out predictions across
folds and computing R², MSE, and RMSE over the full pooled set.  This is
equivalent to the overall prediction accuracy if the model were deployed on
all formulations in the dataset simultaneously.

PER-FOLD SUMMARY
----------------
In addition to the pooled metrics, ``run_loo`` now collects a per-fold
breakdown (``SectionResult.fold_summary``) that records what was left out
in each fold and the fold-level R².  For mix/batch LOO each fold covers
multiple samples so fold R² is meaningful; for row LOO (one sample per
fold) R² is set to NaN and only RMSE is reported.

This is saved as ``loo_fold_summary.csv`` alongside the standard outputs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

from ..config import Config
from ..features import engineer_composition_features
from ..pipeline import make_preprocessor, r2_mse_rmse
from .results import SectionResult


def _feature_cols(df: pd.DataFrame, cfg: Config) -> List[str]:
    """Return composition feature columns present in ``df``."""
    return [c for c in cfg.data.composition_cols if c in df.columns]


def _run_one_target(
    physics_all: pd.DataFrame,
    target_col: str,
    cfg: Config,
    leave_out_unit: str,
) -> Tuple[dict, pd.DataFrame]:
    """Run the full LOO loop for one (physics model, target) combination.

    For each fold, the function:
      1. Splits the data into train/test folds using the specified leave-out
         strategy.
      2. Applies feature engineering, SHAP-guided exclusions, and preprocessing
         to the fold's training data, then transforms the test fold with the
         same fitted preprocessor.
      3. Applies the per-target transform (``cfg.xgb.per_target_transforms``
         takes priority over the global ``cfg.xgb.target_transform``).
      4. Trains an XGBoost model with the fixed hyperparameters from the config.
      5. Predicts on the held-out fold and inverts the target transform.
      6. Stores (y_true, y_pred, fold_id, row_index) for every held-out row.

    Parameters
    ----------
    physics_all:
        Combined train+test physics DataFrame for one model.  LOO uses all
        available data; the train/test split is not used here.
    target_col:
        The column to predict in this LOO run.
    cfg:
        Experiment configuration (fixed XGB params, feature settings, etc.).
    leave_out_unit:
        ``'row'``, ``'mix'``, or ``'batch'``.

    Returns
    -------
    (summary_dict, fold_predictions_df)
        ``summary_dict`` contains aggregated metrics and metadata.
        ``fold_predictions_df`` has one row per held-out sample with
        columns [fold, row_index, mix_id, source_batch, y_true, y_pred].
    """
    feature_cols = _feature_cols(physics_all, cfg)

    # Choose the splitting strategy based on the leave-out unit.
    if leave_out_unit == "mix" and cfg.data.group_column in physics_all.columns:
        # Group-level LOO: all replicates of one unique mix are held out together.
        # This is the more realistic evaluation for predicting new formulations.
        groups = physics_all[cfg.data.group_column].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(physics_all, groups=groups)
        n_folds = int(np.unique(groups).size)
    elif leave_out_unit == "batch" and "Source_Batch" in physics_all.columns:
        # Batch-level LOO: two folds — train on Batch_1/test on Batch_2, then
        # train on Batch_2/test on Batch_1.  The most demanding evaluation:
        # checks cross-batch transferability of learned composition relationships.
        groups = physics_all["Source_Batch"].astype(str).to_numpy()
        splitter = LeaveOneGroupOut()
        split_iter = splitter.split(physics_all, groups=groups)
        n_folds = int(np.unique(groups).size)
    else:
        # Row-level LOO: one measurement row is held out per fold.
        splitter = LeaveOneOut()
        split_iter = splitter.split(physics_all)
        n_folds = int(len(physics_all))

    fold_predictions: List[dict] = []
    use_gpr = getattr(cfg.loo, "model", "xgb") == "gpr"

    # XGBoost-only setup: fixed params and per-target transform.
    if not use_gpr:
        xgb_params = dict(cfg.xgb.fixed_params)
        xgb_params["random_state"] = cfg.seed
        xgb_params["objective"] = cfg.xgb.objective
        xgb_params.update(cfg.xgb.objective_params)
        transform_mode = (
            cfg.xgb.per_target_transforms.get(target_col) or cfg.xgb.target_transform
        )

    for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_fold = physics_all.iloc[train_idx]
        test_fold = physics_all.iloc[test_idx]
        y_test = test_fold[target_col].to_numpy()

        if use_gpr:
            # GPR handles all feature engineering and preprocessing internally.
            from ..gpr import fit_and_evaluate_gpr  # lazy import to avoid circularity
            _, preds_df, _ = fit_and_evaluate_gpr(train_fold, test_fold, target_col, cfg)
            y_pred = preds_df["y_pred"].to_numpy()
        else:
            X_train_raw = train_fold[feature_cols].copy()
            X_test_raw = test_fold[feature_cols].copy()
            y_train = train_fold[target_col].to_numpy()

            if cfg.features.use_engineered:
                X_train = engineer_composition_features(X_train_raw)
                X_test = engineer_composition_features(X_test_raw)
            else:
                X_train = X_train_raw
                X_test = X_test_raw

            # Apply SHAP-guided feature exclusions — mirrors the baseline XGBoost path.
            exclude = cfg.features.exclude_cols or []
            if exclude:
                X_train = X_train.drop(columns=[c for c in exclude if c in X_train.columns])
                X_test = X_test.drop(columns=[c for c in exclude if c in X_test.columns])

            # Fit the preprocessor on this fold's training data only — never
            # on the held-out test fold.  This prevents the OHE category list
            # and imputation statistics from seeing test data.
            preprocessor = make_preprocessor(
                X_train,
                log_numeric_features=cfg.features.log_numeric,
                poly_degree=cfg.features.poly_degree,
            )
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            # Target transform: compute the shift from training data only so that
            # the shift value cannot leak information about the test fold's targets.
            target_shift = 0.0
            if transform_mode == "log1p_nonnegative":
                min_train = float(np.min(y_train))
                if min_train <= -1.0:
                    target_shift = abs(min_train) + 1.001
                y_train_model = np.log1p(y_train + target_shift)
            else:
                y_train_model = y_train

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train_proc, y_train_model, verbose=False)
            y_pred_raw = model.predict(X_test_proc)

            # Invert the target transform to get predictions on the original scale.
            if transform_mode == "log1p_nonnegative":
                y_pred = np.expm1(y_pred_raw) - target_shift
            else:
                y_pred = y_pred_raw

        # Record each held-out row's prediction.  mix_id and source_batch are
        # stored so per-mix and per-batch residual analysis is possible from
        # the saved predictions CSV without re-running the experiment.
        mix_ids = test_fold[cfg.data.group_column].tolist() if cfg.data.group_column in test_fold.columns else [None] * len(test_idx)
        batch_ids = test_fold["Source_Batch"].tolist() if "Source_Batch" in test_fold.columns else [None] * len(test_idx)
        for local_idx, original_idx in enumerate(test_idx):
            fold_predictions.append(
                {
                    "fold": fold_id,
                    "row_index": int(original_idx),
                    "mix_id": mix_ids[local_idx],
                    "source_batch": batch_ids[local_idx],
                    "y_true": float(y_test[local_idx]),
                    "y_pred": float(y_pred[local_idx]),
                }
            )

    fold_pred_df = pd.DataFrame(fold_predictions)
    # Pool all held-out predictions and compute a single set of metrics.
    # This is equivalent to computing metrics as if the model had made one
    # prediction per row across the full dataset.
    r2, mse, rmse = r2_mse_rmse(fold_pred_df["y_true"].to_numpy(), fold_pred_df["y_pred"].to_numpy())
    summary = {
        "target": target_col,
        "n_folds": n_folds,
        "leave_out_unit": leave_out_unit,
        "test_r2": float(r2),
        "test_mse": float(mse),
        "test_rmse": float(rmse),
    }
    return summary, fold_pred_df


def _fold_summary(fold_pred_df: pd.DataFrame, leave_out_unit: str) -> pd.DataFrame:
    """Compute per-fold metrics from pooled fold predictions.

    For mix/batch LOO each fold may contain multiple samples (replicates of
    the same formulation, or all formulations in one batch), so per-fold R²
    is meaningful.  For row LOO (one sample per fold) R² is undefined and
    set to NaN; only RMSE is reported.

    Parameters
    ----------
    fold_pred_df:
        Output of ``_run_one_target``: one row per held-out sample.
    leave_out_unit:
        ``'row'``, ``'mix'``, or ``'batch'``.  Controls which column is used
        as the ``left_out`` label.

    Returns
    -------
    DataFrame with columns:
        fold, left_out, n_samples, test_r2, test_mse, test_rmse
    """
    rows = []
    for fold_id, group in fold_pred_df.groupby("fold"):
        y_true = group["y_true"].to_numpy()
        y_pred = group["y_pred"].to_numpy()
        n = len(group)

        if leave_out_unit == "mix":
            left_out = str(group["mix_id"].iloc[0])
        elif leave_out_unit == "batch":
            left_out = str(group["source_batch"].iloc[0])
        else:
            left_out = str(group["row_index"].iloc[0])

        if n > 1:
            r2, mse, rmse = r2_mse_rmse(y_true, y_pred)
        else:
            diff = float(y_true[0] - y_pred[0])
            r2 = float("nan")
            mse = float(diff ** 2)
            rmse = float(abs(diff))

        rows.append({
            "fold": int(fold_id),
            "left_out": left_out,
            "n_samples": n,
            "test_r2": r2,
            "test_mse": mse,
            "test_rmse": rmse,
        })
    return pd.DataFrame(rows)


def run_loo(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]], cfg: Config
) -> SectionResult:
    """Run all configured LOO variants for every physics model.

    Iterates over the LOO variants defined in ``cfg.loo.variants`` (typically
    'row' and 'mix'), then over each physics model, then over the target
    columns selected by ``cfg.loo.target_scope``.

    For each combination, ``_run_one_target`` runs the full fold loop and
    returns per-fold predictions and aggregate metrics.  Results are collected
    and assembled into a single SectionResult containing:
      - ``model_summary``: one row per (variant, model), metrics averaged over
        the selected targets.
      - ``per_target``: one row per (variant, model, target).
      - ``predictions``: all fold-level (y_true, y_pred) records.
      - ``fold_summary``: per-fold breakdown showing what was left out in each
        fold and the fold-level R² (NaN for row LOO, meaningful for mix/batch).

    Parameters
    ----------
    physics_datasets:
        Nested dict from ``build_all_physics_datasets``.  Train and test are
        concatenated inside this function because LOO uses all available data.
    cfg:
        Experiment configuration.

    Returns
    -------
    SectionResult named ``'loo'``.
    """
    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_preds: List[pd.DataFrame] = []
    all_fold_summaries: List[pd.DataFrame] = []

    for variant in cfg.loo.variants:
        print(f"\n{'='*70}\nLOO variant: {variant.id} (leave_out_unit={variant.leave_out_unit})\n{'='*70}")
        for model_key in cfg.physics.models:
            display_name = cfg.display_name(model_key)
            # Concatenate train and test splits so LOO has access to all rows.
            # This is correct for LOO: the split into folds is handled internally
            # by the splitter, so using only the train split would under-utilise
            # the data.
            physics_all = pd.concat(
                [physics_datasets[model_key]["train"], physics_datasets[model_key]["test"]],
                ignore_index=True,
            )
            target_cols = cfg.targets_for(model_key)
            # 'first' scope runs only on the primary target (e.g. eta_0) to
            # reduce runtime during exploratory experiments.
            targets_to_run = target_cols[:1] if cfg.loo.target_scope == "first" else target_cols

            model_rows: List[dict] = []
            for target_col in targets_to_run:
                summary, fold_preds = _run_one_target(
                    physics_all, target_col, cfg, variant.leave_out_unit
                )
                summary.update(
                    {
                        "model_name": display_name,
                        "model_key": model_key,
                        "variant_id": variant.id,
                        "leave_out_unit": variant.leave_out_unit,
                    }
                )
                per_target_rows.append(summary)
                model_rows.append(summary)

                all_preds.append(
                    fold_preds.assign(
                        model_name=display_name,
                        model_key=model_key,
                        target=target_col,
                        variant_id=variant.id,
                    )
                )

                # Per-fold breakdown: records what was left out and fold-level R².
                fs = _fold_summary(fold_preds, variant.leave_out_unit)
                fs = fs.assign(
                    model_name=display_name,
                    model_key=model_key,
                    target=target_col,
                    variant_id=variant.id,
                    leave_out_unit=variant.leave_out_unit,
                )
                all_fold_summaries.append(fs)

            if model_rows:
                model_summary_rows.append(
                    {
                        "variant_id": variant.id,
                        "leave_out_unit": variant.leave_out_unit,
                        "model_name": display_name,
                        "model_key": model_key,
                        "test_r2": float(np.mean([r["test_r2"] for r in model_rows])),
                        "test_mse": float(np.mean([r["test_mse"] for r in model_rows])),
                        "test_rmse": float(np.mean([r["test_rmse"] for r in model_rows])),
                    }
                )
                mean_r2 = model_summary_rows[-1]["test_r2"]
                print(f"  {display_name:18s} | mean R²={mean_r2:.4f}")

    fold_summary_df = (
        pd.concat(all_fold_summaries, ignore_index=True)
        if all_fold_summaries
        else None
    )

    return SectionResult(
        name="loo",
        model_summary=pd.DataFrame(model_summary_rows)
        .sort_values(["variant_id", "test_r2"], ascending=[True, False])
        .reset_index(drop=True),
        per_target=pd.DataFrame(per_target_rows),
        predictions=pd.concat(all_preds, ignore_index=True) if all_preds else None,
        fold_summary=fold_summary_df,
    )
