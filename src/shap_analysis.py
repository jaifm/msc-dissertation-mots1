"""SHAP-based feature importance analysis for composition → rheology models.

OVERVIEW
--------
This module computes SHAP (SHapley Additive exPlanations) values for every
(physics model, target) pair using fast XGBoost TreeExplainer.  The goal is
to identify which engineered composition features are informative across
targets and which are consistently near-zero — candidates for removal.

WORKFLOW
--------
1. Call ``run_shap_analysis(physics_datasets, cfg)`` from a notebook cell
   (or after ``build_all_physics_datasets``).
2. Inspect the returned ``ShapResults`` object:
   - ``per_target``: mean |SHAP| per feature for every (model, target) pair.
   - ``aggregated``: mean |SHAP| averaged across all targets, sorted descending.
   - ``low_importance``: features whose aggregated mean |SHAP| falls below
     ``prune_threshold`` (default: 5% of the maximum).
3. Pass ``low_importance`` to a new experiment YAML via
   ``features.exclude_cols`` to re-run with the pruned feature set.

WHY FIXED PARAMS?
-----------------
Running a full Optuna search per (model, target) during SHAP analysis would
take as long as a full experiment run, defeating the purpose of a fast
exploratory pass.  Instead, ``run_shap_analysis`` uses ``cfg.xgb.fixed_params``
(the same defaults used by the LOO section) to fit each model quickly.  The
resulting SHAP values are indicative of importance direction and magnitude —
they do not need to come from a fully tuned model to identify which features
are structurally uninformative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .config import Config
from .features import engineer_composition_features
from .pipeline import make_preprocessor
from .tuning import _apply_target_transform, _select_feature_cols


@dataclass
class ShapResults:
    """SHAP importance summary from a single analysis run.

    Attributes
    ----------
    per_target:
        DataFrame with columns [model_key, target, feature, mean_abs_shap].
        One row per (model, target, feature) combination.
    aggregated:
        DataFrame with columns [feature, mean_abs_shap], sorted descending.
        Averaged over all (model, target) pairs weighted equally.
    low_importance:
        List of feature names whose aggregated mean |SHAP| is below
        ``prune_threshold * max(mean_abs_shap)``.  These are candidates for
        exclusion in a follow-up experiment.
    feature_names:
        Ordered list of all feature names seen during analysis (post-OHE).
    prune_threshold:
        The relative threshold used to classify low-importance features.
    """
    per_target: pd.DataFrame
    aggregated: pd.DataFrame
    low_importance: List[str]
    feature_names: List[str]
    prune_threshold: float


def run_shap_analysis(
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]],
    cfg: Config,
    *,
    prune_threshold: float = 0.05,
    model_keys: Optional[List[str]] = None,
) -> ShapResults:
    """Compute SHAP feature importance for all (model, target) pairs.

    Uses ``cfg.xgb.fixed_params`` (same as the LOO section defaults) for a
    fast, non-tuned XGBoost fit so the analysis runs in seconds rather than
    minutes.

    Parameters
    ----------
    physics_datasets:
        Output of ``build_all_physics_datasets``.
    cfg:
        Experiment configuration.  ``cfg.xgb.fixed_params`` and
        ``cfg.features`` control preprocessing.
    prune_threshold:
        Features with mean |SHAP| < ``prune_threshold * max(mean |SHAP|)``
        are added to ``ShapResults.low_importance``.  Default 0.05 (5%).
    model_keys:
        Subset of physics models to analyse.  Defaults to all models in
        ``cfg.physics.models``.

    Returns
    -------
    ShapResults
    """
    keys = model_keys if model_keys is not None else cfg.physics.models
    rows: List[dict] = []
    all_feature_names: List[str] = []

    for model_key in keys:
        display = cfg.display_name(model_key)
        train_p = physics_datasets[model_key]["train"]
        targets = train_p.attrs["target_cols"]

        feature_cols = _select_feature_cols(train_p, cfg)
        X_raw = train_p[feature_cols].copy()
        if cfg.features.use_engineered:
            X = engineer_composition_features(X_raw)
        else:
            X = X_raw

        # Apply exclude_cols if configured.
        exclude = getattr(cfg.features, "exclude_cols", []) or []
        if exclude:
            X = X.drop(columns=[c for c in exclude if c in X.columns])

        preprocessor = make_preprocessor(
            X,
            log_numeric_features=cfg.features.log_numeric,
            poly_degree=cfg.features.poly_degree,
        )
        X_proc = preprocessor.fit_transform(X)

        # Recover feature names after OHE expansion.
        try:
            feat_names = preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            feat_names = [f"f{i}" for i in range(X_proc.shape[1])]

        if not all_feature_names:
            all_feature_names = feat_names

        params = {
            **cfg.xgb.fixed_params,
            "objective": cfg.xgb.objective,
            **cfg.xgb.objective_params,
            "random_state": cfg.seed,
        }

        for target_col in targets:
            y = train_p[target_col].to_numpy()
            transform_mode = (
                cfg.xgb.per_target_transforms.get(target_col) or cfg.xgb.target_transform
            )
            y_model, shift = _apply_target_transform(y, transform_mode)

            model = xgb.XGBRegressor(**params)
            model.fit(X_proc, y_model)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_proc)  # shape (n_samples, n_features)
            mean_abs = np.abs(shap_values).mean(axis=0)

            for fname, mabs in zip(feat_names, mean_abs):
                rows.append({
                    "model_key": model_key,
                    "model_name": display,
                    "target": target_col,
                    "feature": fname,
                    "mean_abs_shap": float(mabs),
                })

        print(f"  {display}: {len(targets)} targets done")

    per_target = pd.DataFrame(rows)

    # Aggregate: simple mean across all (model, target) pairs.
    aggregated = (
        per_target.groupby("feature")["mean_abs_shap"]
        .mean()
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    max_shap = aggregated["mean_abs_shap"].max()
    cutoff = prune_threshold * max_shap
    low_imp = aggregated.loc[
        aggregated["mean_abs_shap"] < cutoff, "feature"
    ].tolist()

    return ShapResults(
        per_target=per_target,
        aggregated=aggregated,
        low_importance=low_imp,
        feature_names=all_feature_names,
        prune_threshold=prune_threshold,
    )


def print_importance_table(results: ShapResults, top_n: int = 20) -> None:
    """Print a ranked feature importance table to stdout."""
    print(f"\n{'Rank':>4}  {'Feature':<35}  {'Mean |SHAP|':>12}  {'% of max':>9}")
    print("-" * 68)
    max_s = results.aggregated["mean_abs_shap"].max()
    for i, row in results.aggregated.head(top_n).iterrows():
        pct = 100.0 * row["mean_abs_shap"] / max_s
        tag = " ← LOW" if row["feature"] in results.low_importance else ""
        print(f"{i+1:>4}  {row['feature']:<35}  {row['mean_abs_shap']:>12.4f}  {pct:>8.1f}%{tag}")
    remaining = len(results.aggregated) - top_n
    if remaining > 0:
        low_in_rest = sum(
            1 for f in results.low_importance
            if f not in results.aggregated.head(top_n)["feature"].values
        )
        print(f"       ... {remaining} more features ({low_in_rest} low-importance)")
    print(f"\nLow-importance threshold: {results.prune_threshold*100:.0f}% of max")
    print(f"Features to prune ({len(results.low_importance)}):")
    for f in results.low_importance:
        print(f"  - {f}")
