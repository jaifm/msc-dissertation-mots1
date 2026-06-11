"""Section runner for the physics-augmented training methodology.

Single public entry point: ``run_physics_augmented(train_df, test_df, cfg, run_id)``.
The runner dispatches to this function when ``'physics_augmented'`` is in
``cfg.run.sections``.

For every (physics oracle, learner) pair in the configuration:

  1. Build the long-format augmented training set + measured-only test
     set via ``data.build_augmented_split``.
  2. Fit the chosen learner.
  3. Score per measurement shear rate on the test set.
  4. Append rows to the section's ``model_summary``, ``per_target``, and
     ``predictions`` tables in the schema the rest of the pipeline uses.

The output ``SectionResult`` is named ``'physics_augmented'`` and conforms
to the standard schema so downstream notebook cells render it without
modification.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import Config
from ..experiments.results import SectionResult
from .augment import SUPPORTED_MODELS as _SUPPORTED_PHYSICS
from .data import (
    AugmentedSplit,
    build_augmented_split,
    feature_matrix,
    target_array,
    weight_array,
)
from .learners import (
    SUPPORTED_LEARNERS as _SUPPORTED_LEARNERS,
    fit_learner,
)


def _format_stats(stats) -> str:
    return (
        f"train_rows={stats.n_train_rows} "
        f"fit_ok={stats.n_fit_success} "
        f"skipped_nan_viscosity={stats.n_skipped_nan_viscosity} "
        f"skipped_nan_composition={stats.n_skipped_nan_composition} "
        f"skipped_invalid_fit={stats.n_skipped_invalid_fit}"
    )


def run_physics_augmented(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    *,
    run_id: str = "",
) -> SectionResult:
    """Run every (physics oracle × learner) combination requested in cfg.

    The cfg block consumed here is ``cfg.physics_augmented`` (see
    ``src.config.PhysicsAugmentedCfg``).  Defaults documented there.
    """
    pa_cfg = cfg.physics_augmented  # type: ignore[attr-defined]
    seed = int(pa_cfg.seed if pa_cfg.seed is not None else cfg.seed)

    physics_models = [m for m in pa_cfg.models if m in _SUPPORTED_PHYSICS]
    bad_phys = set(pa_cfg.models) - set(_SUPPORTED_PHYSICS)
    if bad_phys:
        raise ValueError(
            f"Unknown physics_augmented.models: {sorted(bad_phys)}; "
            f"supported: {_SUPPORTED_PHYSICS}"
        )
    learners = [l for l in pa_cfg.learners if l in _SUPPORTED_LEARNERS]
    bad_learn = set(pa_cfg.learners) - set(_SUPPORTED_LEARNERS)
    if bad_learn:
        raise ValueError(
            f"Unknown physics_augmented.learners: {sorted(bad_learn)}; "
            f"supported: {_SUPPORTED_LEARNERS}"
        )

    measurement_shears = list(map(float, cfg.physics.viscosity_shears))

    print(
        f"\n{'='*70}\n"
        f"Physics-Augmented training\n"
        f"{'='*70}\n"
        f"  Physics oracles  : {physics_models}\n"
        f"  Learners         : {learners}\n"
        f"  n_aug_points     : {pa_cfg.n_aug_points}\n"
        f"  Shear range      : {pa_cfg.shear_range}\n"
        f"  Log-uniform γ̇    : {pa_cfg.log_uniform_gamma}\n"
        f"  Measured weight  : {pa_cfg.measured_weight}\n"
        f"  Target transform : {pa_cfg.target_transform}"
    )

    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_predictions: List[pd.DataFrame] = []
    augmented_train_dfs: Dict[str, pd.DataFrame] = {}

    for physics_key in physics_models:
        print(f"\n--- Physics oracle: {physics_key} ---")
        split = build_augmented_split(
            train_df=train_df,
            test_df=test_df,
            cfg=cfg,
            model_key=physics_key,
            n_aug_points=int(pa_cfg.n_aug_points),
            shear_range=tuple(pa_cfg.shear_range),
            log_uniform=bool(pa_cfg.log_uniform_gamma),
            measured_weight=float(pa_cfg.measured_weight),
            seed=seed,
        )
        print(f"  Augmentation stats: {_format_stats(split.stats)}")
        print(
            f"  Long-format frames: train={len(split.train_long)}, "
            f"test={len(split.test_long)}, features={len(split.feature_cols)}"
        )
        augmented_train_dfs[f"physics_augmented_{physics_key}_train_long.csv"] = (
            split.train_long
        )

        X_train = feature_matrix(split.train_long, split.feature_cols)
        y_train = target_array(split.train_long)
        w_train = weight_array(split.train_long)
        X_test = feature_matrix(split.test_long, split.feature_cols)

        for learner_name in learners:
            print(f"  -> Learner: {learner_name}")
            result = fit_learner(
                learner_name,
                X_train=X_train,
                y_train=y_train,
                weights_train=w_train,
                X_test=X_test,
                test_long=split.test_long,
                measurement_shears=measurement_shears,
                target_transform=pa_cfg.target_transform,
                cfg=cfg,
                seed=seed,
            )

            variant_id = f"{physics_key}__{learner_name}"
            # Per-shear rows.
            for _, r in result.per_shear.iterrows():
                per_target_rows.append({
                    "variant": variant_id,
                    "physics_oracle": physics_key,
                    "learner": learner_name,
                    "target": f"Viscosity_at_shear_rate_{int(r['shear_rate'])}_1/s",
                    "shear_rate": float(r["shear_rate"]),
                    "n": int(r["n"]),
                    "test_r2": float(r["r2"]),
                    "test_mae": float(r["mae"]),
                    "test_rmse": float(r["rmse"]),
                    "model_name": variant_id,
                    "model_key": variant_id,
                })

            # Overall (pooled across shears) summary row.
            model_summary_rows.append({
                "variant": variant_id,
                "physics_oracle": physics_key,
                "learner": learner_name,
                "model_name": variant_id,
                "model_key": variant_id,
                "test_r2": float(result.overall["r2"]),
                "test_r2_log": float(result.overall["r2_log"]),
                "test_mae": float(result.overall["mae"]),
                "test_rmse": float(result.overall["rmse"]),
                "n_test": int(result.overall["n"]),
                "n_aug_points": int(pa_cfg.n_aug_points),
                "physics_model": physics_key,
            })

            # Long-format predictions for downstream plots.
            preds_df = split.test_long[["row_id", "shear_rate", "viscosity"]].copy()
            preds_df = preds_df.rename(columns={"viscosity": "y_true"})
            preds_df["y_pred"] = np.asarray(result.y_pred_test, float)
            preds_df["target"] = preds_df["shear_rate"].apply(
                lambda g: f"Viscosity_at_shear_rate_{int(g)}_1/s"
            )
            preds_df["variant"] = variant_id
            preds_df["physics_oracle"] = physics_key
            preds_df["learner"] = learner_name
            preds_df["model_name"] = variant_id
            preds_df["model_key"] = variant_id
            all_predictions.append(preds_df)

            print(
                f"     R²(raw)={result.overall['r2']:.3f}  "
                f"R²(log)={result.overall['r2_log']:.3f}  "
                f"RMSE={result.overall['rmse']:.3f}  "
                f"MAE={result.overall['mae']:.3f}"
            )
            for _, r in result.per_shear.iterrows():
                print(
                    f"       γ̇={r['shear_rate']:>6.1f}  "
                    f"R²={r['r2']:.3f}  RMSE={r['rmse']:.3f}  MAE={r['mae']:.3f}"
                )

    model_summary_df = pd.DataFrame(model_summary_rows)
    per_target_df = pd.DataFrame(per_target_rows)
    predictions_df = (
        pd.concat(all_predictions, axis=0, ignore_index=True)
        if all_predictions else None
    )

    return SectionResult(
        name="physics_augmented",
        model_summary=model_summary_df,
        per_target=per_target_df,
        predictions=predictions_df,
        fold_summary=None,
        curve_summary=None,
        additional_dfs=augmented_train_dfs if augmented_train_dfs else None,
    )
