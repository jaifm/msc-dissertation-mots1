"""Gaussian Process Regression training and evaluation for rheology parameter prediction.

OVERVIEW
--------
This module provides ``fit_and_evaluate_gpr``, a drop-in replacement for
``tuning.tune_and_evaluate_xgb`` that uses a Gaussian Process instead of
XGBoost.  Both functions accept the same arguments and return the same
(result_dict, predictions_df, train_predictions) tuple, so a single section
runner can drive either model without branching.

WHY GPR?
--------
XGBoost is a discriminative model that minimises a point-estimate loss.  GPR
is a Bayesian non-parametric model that places a distribution over functions.
For the dissertation comparison, GPR offers:

  1. **Principled uncertainty**: the posterior standard deviation quantifies
     which compositions the model is uncertain about — useful for identifying
     formulations that are poorly supported by the 120-row training set.

  2. **Exact kernel hyperparameter tuning**: length scales, output variance,
     and noise level are all optimised analytically by maximising the log
     marginal likelihood (no Optuna budget required).

  3. **Smooth interpolation**: the Matern kernel interpolates smoothly between
     training points, which suits the continuous composition → parameter
     relationship better than the step-function boundaries produced by trees.

PREPROCESSING DIFFERENCE
-------------------------
The XGBoost preprocessor (``pipeline.make_preprocessor``) does not standardise
numeric features, because tree-based models are scale-invariant.  GPR kernels
measure sample similarity via Euclidean distance in feature space; without
standardisation, a large-magnitude feature like Solid_Content_pct (~70) would
dominate the learned length scale, making low-magnitude features like
KS6L_pct (~0.5) effectively invisible.  ``_make_gpr_preprocessor`` adds a
``StandardScaler`` step after imputation.

KERNEL DESIGN
-------------
Both kernels follow the pattern:

    ConstantKernel × BaseKernel + WhiteKernel

The ``ConstantKernel`` scales the prior covariance amplitude.  The
``WhiteKernel`` models observation noise — without it the GPR fits the
training data exactly (zero noise assumption), producing overconfident
uncertainty estimates and poor generalisation on noisy rheology measurements.

'matern25' (default) uses Matern(ν=2.5), which is once mean-square
differentiable.  This is softer than RBF (infinitely differentiable) and
better suited to real-world data where the composition–parameter relationship
may have local irregularities.

TARGET TRANSFORMS
-----------------
The same transform modes as ``tuning.py`` are supported (``none``,
``log1p_nonnegative``) via ``cfg.gpr.target_transform``.  With
``normalize_y=True``, GPR also internally centres and scales the (possibly
transformed) target before kernel computations, improving numerical
conditioning.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    DotProduct,
    Matern,
    RBF,
    WhiteKernel,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .features import engineer_composition_features
from .pipeline import r2_mse_rmse
from .tuning import (
    _apply_target_transform,
    _invert_target_transform,
    _select_feature_cols,
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _make_gpr_preprocessor(X_df: pd.DataFrame) -> Pipeline:
    """Build a preprocessor that one-hot-encodes categoricals, imputes
    numerics, then applies a global StandardScaler to ALL outputs.

    The global StandardScaler (applied after the ColumnTransformer) ensures
    that both numeric and OHE-binary features contribute equally to the
    kernel's L2 distance metric.  Without it, OHE columns have std ≈ 0.3–0.5
    while numeric columns have std = 1.0 after per-branch scaling, causing
    the kernel to systematically down-weight dispersant-type information.
    """
    numeric_cols = [
        c for c in X_df.columns
        if is_numeric_dtype(X_df[c]) and not is_bool_dtype(X_df[c])
    ]
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    transformers = []
    if categorical_cols:
        transformers.append((
            "cat",
            OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
            categorical_cols,
        ))
    if numeric_cols:
        # Impute only — the global StandardScaler below handles scaling for
        # all branches uniformly, avoiding double-standardisation of numerics.
        transformers.append((
            "num",
            SimpleImputer(strategy="median"),
            numeric_cols,
        ))
    ct = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("ct", ct), ("scale", StandardScaler())])


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------


def _build_kernel(kernel_name: str, n_features: int = 1, ard: bool = False):
    """Return an sklearn kernel object from a config name string.

    Bounds are set wide (1e-3, 1e3 for length scales) so the log marginal
    likelihood optimiser can freely explore, with the likelihood itself
    acting as the regulariser.

    Parameters
    ----------
    kernel_name:
        ``'matern25'`` or ``'rbf'``.
    n_features:
        Number of input features.  Only used when ``ard=True`` to initialise
        one length scale per feature.
    ard:
        When True, use ARD (Automatic Relevance Determination): the kernel
        receives a length-scale vector of shape ``(n_features,)`` so the
        marginal likelihood can tune each dimension independently.
    """
    ls_bounds = (1e-3, 1e3)
    noise_bounds = (1e-5, 10.0)
    amp_bounds = (1e-3, 1e3)
    # ARD: one length scale per feature; isotropic otherwise.
    ls_init = np.ones(n_features) if (ard and n_features > 1) else 1.0

    if kernel_name == "rbf":
        return (
            ConstantKernel(1.0, constant_value_bounds=amp_bounds)
            * RBF(length_scale=ls_init, length_scale_bounds=ls_bounds)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
        )
    if kernel_name == "matern25":
        return (
            ConstantKernel(1.0, constant_value_bounds=amp_bounds)
            * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
        )
    if kernel_name == "matern15":
        # ν=1.5 is once mean-square differentiable — rougher than ν=2.5.
        # Suited to datasets with sharp compositional transitions (e.g. the
        # ~2 wt% C65 percolation threshold for the conductive network).
        return (
            ConstantKernel(1.0, constant_value_bounds=amp_bounds)
            * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=1.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
        )
    if kernel_name == "matern25_linear":
        # Composite kernel: Matern(ν=2.5) captures smooth nonlinear structure;
        # DotProduct captures the quasi-linear composition → viscosity trend
        # predicted by Krieger-Dougherty theory at low-to-moderate packing.
        # The two components have independent amplitudes so the optimiser can
        # suppress either if it adds no marginal likelihood improvement.
        return (
            ConstantKernel(1.0, constant_value_bounds=amp_bounds)
            * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=2.5)
            + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=noise_bounds)
        )
    raise ValueError(
        f"Unknown GPR kernel: {kernel_name!r}. "
        f"Valid: 'rbf', 'matern25', 'matern15', 'matern25_linear'."
    )


# ---------------------------------------------------------------------------
# Main training / evaluation function
# ---------------------------------------------------------------------------


def fit_and_evaluate_gpr(
    physics_train: pd.DataFrame,
    physics_test: pd.DataFrame,
    target_col: str,
    cfg: Config,
) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    """Fit a GPR model and evaluate on the held-out test set.

    Mirrors the interface of ``tuning.tune_and_evaluate_xgb`` exactly so both
    can be driven by the same section runner.

    Parameters
    ----------
    physics_train, physics_test:
        DataFrames with fitted physics parameters appended (from
        ``build_physics_dataset``).
    target_col:
        Name of the column to predict.
    cfg:
        Experiment configuration.  The ``cfg.gpr`` block controls kernel,
        restarts, and target transform settings.

    Returns
    -------
    (result_dict, predictions_df, train_predictions_array)
        ``result_dict`` contains R², MSE, RMSE, plus the fitted kernel string
        and log marginal likelihood for diagnostics.
        ``predictions_df`` has columns [y_true, y_pred, y_std, target] where
        ``y_std`` is the posterior standard deviation on the test set.
        ``train_predictions_array`` contains model predictions on the training
        set (used for overfitting diagnostics, same role as in tuning.py).
    """
    # --- Feature selection and engineering -----------------------------------
    feature_cols = _select_feature_cols(physics_train, cfg)
    X_train_raw = physics_train[feature_cols].copy()
    X_test_raw = physics_test[feature_cols].copy()
    y_train = physics_train[target_col].to_numpy()
    y_test = physics_test[target_col].to_numpy()

    if cfg.features.use_engineered:
        X_train = engineer_composition_features(X_train_raw)
        X_test = engineer_composition_features(X_test_raw)
    else:
        X_train = X_train_raw
        X_test = X_test_raw

    # Apply SHAP-guided feature exclusions (mirrors the XGBoost path).
    exclude = cfg.features.exclude_cols or []
    if exclude:
        X_train = X_train.drop(columns=[c for c in exclude if c in X_train.columns])
        X_test  = X_test.drop(columns=[c for c in exclude if c in X_test.columns])

    # --- Preprocessing (impute + standardise) --------------------------------
    preprocessor = _make_gpr_preprocessor(X_train)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- Target transform (per-target override takes priority) ---------------
    transform_mode = (
        cfg.gpr.per_target_transforms.get(target_col) or cfg.gpr.target_transform
    )
    y_train_model, shift = _apply_target_transform(y_train, transform_mode)

    # --- Fit GPR -------------------------------------------------------------
    kernel = _build_kernel(
        cfg.gpr.kernel,
        n_features=X_train_proc.shape[1],
        ard=cfg.gpr.ard,
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=cfg.gpr.n_restarts,
        normalize_y=cfg.gpr.normalize_y,
        random_state=cfg.seed,
    )
    gpr.fit(X_train_proc, y_train_model)

    # --- Predict and back-transform ------------------------------------------
    y_train_pred_model, _ = gpr.predict(X_train_proc, return_std=True)
    y_test_pred_model, y_test_std = gpr.predict(X_test_proc, return_std=True)

    y_train_pred = _invert_target_transform(y_train_pred_model, transform_mode, shift)
    y_test_pred = _invert_target_transform(y_test_pred_model, transform_mode, shift)

    # --- Metrics -------------------------------------------------------------
    train_r2, train_mse, train_rmse = r2_mse_rmse(y_train, y_train_pred)
    test_r2, test_mse, test_rmse = r2_mse_rmse(y_test, y_test_pred)

    result = {
        "target": target_col,
        "train_r2": train_r2,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        # Fitted kernel string shows optimised hyperparameter values —
        # useful for verifying that the length scales are physically sensible.
        "kernel_fitted": str(gpr.kernel_),
        "log_marginal_likelihood": float(gpr.log_marginal_likelihood_value_),
        "target_transform": transform_mode,
    }

    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_test_pred,
        # Posterior std in the original (back-transformed) units where possible.
        # For 'none' transform this is exact; for log transforms it is an
        # approximation (first-order delta method) — use for ranking, not
        # quantitative calibration.
        "y_std": y_test_std,
        "target": target_col,
    })

    return result, predictions_df, y_train_pred
