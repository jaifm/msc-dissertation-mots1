"""Build (composition, log10 γ̇) → η long-format datasets for training.

This module takes the existing project train/test splits and produces:

  * ``X_train, y_train, weights_train`` — one row per ``(composition, γ̇)``
    pair (measured + augmented).  The sample-weight column lets the
    learner up-weight the three measured points relative to the augmented
    ones, which is a deliberate knob — augmented samples are physics-
    interpolated, not ground truth, and the user may want them to act as
    soft regularisation rather than first-class training data.

  * ``X_test, y_test, row_ids_test`` — one row per ``(composition, γ̇)``
    pair from the test split's three measured viscosities only.  ``row_ids_test``
    aligns predictions back to the source row so per-row diagnostics can
    be computed downstream.

The feature column ``log_gamma`` is always added (the user explicitly chose
log-scale shear as the ML feature).  The original ``shear_rate`` column is
preserved alongside so per-shear-rate scoring stays straightforward.

The composition feature set itself reuses the main pipeline's feature
engineering when requested (via ``cfg.features``), keeping the
physics-augmented experiment directly comparable to baseline-section
results on the same engineered feature set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..features import add_crowding_feature, engineer_composition_features
from .augment import (
    AugmentationStats,
    SUPPORTED_MODELS,
    augment_train_rows,
    make_test_rows,
)

# Reserved bookkeeping columns added to the long-format frame.  These are
# excluded from the X feature matrix at the learner boundary.
_BOOKKEEPING_COLS = ("row_id", "source_row_idx", "shear_rate", "viscosity",
                     "source", "physics_model", "sample_weight", "log_gamma")


@dataclass
class AugmentedSplit:
    """All the bits the section runner needs for one (physics_model) run.

    Attributes
    ----------
    train_long, test_long:
        Long-format DataFrames with feature columns plus the bookkeeping
        columns listed in ``_BOOKKEEPING_COLS``.
    feature_cols:
        Column names of the X feature matrix (composition + log_gamma).
    target_col:
        ``'viscosity'``.  Kept as a constant for callers that prefer not
        to hard-code the literal.
    weight_col:
        ``'sample_weight'``.
    stats:
        Augmentation diagnostics — useful for the section's stdout summary.
    """
    train_long: pd.DataFrame
    test_long: pd.DataFrame
    feature_cols: List[str]
    target_col: str
    weight_col: str
    stats: AugmentationStats


def _composition_cols(cfg: Config) -> List[str]:
    """Return the composition column names to carry into the long frames.

    Mirrors what the main pipeline considers the raw composition: every
    column listed in ``cfg.data.composition_cols``.  Engineered features
    are appended later, after augmentation, so the physics fitter only
    ever sees the raw composition columns (which already include solid
    content, dispersant type, etc).
    """
    return list(cfg.data.composition_cols)


def _maybe_engineer(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Apply ``cfg.features.use_engineered`` to a wide-format DataFrame.

    Returns ``(engineered_df, engineered_col_names)``.  When engineering
    is disabled the original df and the original columns are returned
    unchanged.
    """
    _non_feature = {"Composite_Mix_ID", "Source_Batch", "Formulation"}
    phi_max = getattr(cfg.features, "crowding_phi_max", None)
    if not cfg.features.use_engineered:
        # The Maron–Pierce crowding term is independent of the full
        # engineered set so targeted ablations can add it on its own.
        if phi_max is not None:
            df = add_crowding_feature(df, phi_max)
        return df, [c for c in df.columns
                    if c not in _non_feature
                    and c not in cfg.data.viscosity_columns]
    engineered = engineer_composition_features(df)
    if phi_max is not None:
        engineered = add_crowding_feature(engineered, phi_max)
    # Drop columns the user asked to exclude (SHAP-pruned set, etc.).
    excl = set(getattr(cfg.features, "exclude_cols", []) or [])
    engineered = engineered.drop(columns=[c for c in excl if c in engineered.columns])
    feat_cols = [c for c in engineered.columns
                 if c not in _non_feature
                 and c not in cfg.data.viscosity_columns]
    return engineered, feat_cols


def build_augmented_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    *,
    model_key: str,
    n_aug_points: int,
    shear_range: Tuple[float, float],
    log_uniform: bool,
    measured_weight: float,
    seed: int,
) -> AugmentedSplit:
    """Build train/test long-format frames for one physics-augmented run.

    Parameters
    ----------
    train_df, test_df:
        Wide-format splits as loaded by ``data.splits.load_splits``.
    cfg:
        Project configuration — drives the feature engineering choice and
        provides the viscosity column names + measurement shears.
    model_key:
        ``'power_law'`` or ``'constrained_cy'``.  The physics oracle used
        to interpolate the augmented samples.
    n_aug_points:
        How many augmentation samples per train row.
    shear_range:
        Tuple ``(γ̇_min, γ̇_max)`` for the augmentation sampling grid.
    log_uniform:
        When True, samples log-uniformly across ``shear_range``.
    measured_weight:
        Sample weight applied to measured rows.  Augmented rows always
        get weight 1.0.  A value of 1.0 means equal weighting; higher
        values up-weight the real measurements so the model gives them
        priority over the physics-interpolated points.
    seed:
        Reproducibility seed for the augmentation sampler.
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown physics_augmented model {model_key!r}; "
            f"supported: {SUPPORTED_MODELS}"
        )

    measurement_shears = list(map(float, cfg.physics.viscosity_shears))

    # Engineer features on the wide DataFrames first; the long-format
    # builder then carries the engineered columns through unchanged.
    train_eng, train_feat_cols = _maybe_engineer(train_df.copy(), cfg)
    test_eng, test_feat_cols = _maybe_engineer(test_df.copy(), cfg)

    # Use the intersection of train and test feature columns to guard
    # against any mismatch from a column being introduced by only one
    # side's engineering pass.
    composition_cols = [c for c in train_feat_cols if c in test_feat_cols]

    train_long, stats = augment_train_rows(
        train_eng,
        model_key=model_key,
        viscosity_columns=cfg.data.viscosity_columns,
        measurement_shears=measurement_shears,
        composition_cols=composition_cols,
        n_aug_points=n_aug_points,
        shear_range=shear_range,
        log_uniform=log_uniform,
        seed=seed,
    )
    test_long = make_test_rows(
        test_eng,
        viscosity_columns=cfg.data.viscosity_columns,
        measurement_shears=measurement_shears,
        composition_cols=composition_cols,
    )

    # log10 γ̇ as the model feature (the user explicitly chose log scale).
    train_long["log_gamma"] = np.log10(train_long["shear_rate"].astype(float))
    test_long["log_gamma"] = np.log10(test_long["shear_rate"].astype(float))

    # Sample weights: measured rows get ``measured_weight``, augmented get 1.0.
    train_long["sample_weight"] = np.where(
        train_long["source"] == "measured",
        float(measured_weight),
        1.0,
    )
    test_long["sample_weight"] = 1.0

    feature_cols = list(composition_cols) + ["log_gamma"]
    return AugmentedSplit(
        train_long=train_long.reset_index(drop=True),
        test_long=test_long.reset_index(drop=True),
        feature_cols=feature_cols,
        target_col="viscosity",
        weight_col="sample_weight",
        stats=stats,
    )


def feature_matrix(long_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Return only the feature columns from a long-format frame."""
    missing = [c for c in feature_cols if c not in long_df.columns]
    if missing:
        raise KeyError(f"Long frame is missing feature columns: {missing}")
    return long_df[list(feature_cols)].copy()


def target_array(long_df: pd.DataFrame) -> np.ndarray:
    return long_df["viscosity"].to_numpy(dtype=float)


def weight_array(long_df: pd.DataFrame) -> np.ndarray:
    if "sample_weight" not in long_df.columns:
        return np.ones(len(long_df), dtype=float)
    return long_df["sample_weight"].to_numpy(dtype=float)
