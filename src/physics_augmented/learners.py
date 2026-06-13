"""XGBoost and GPR student learners for the physics-augmented training set.

These wrappers are deliberately thin.  Both consume the long-format
``(composition, log γ̇)`` feature matrices produced by ``data.py`` and
return per-prediction arrays plus the standard metric dict.

The wrappers reuse the existing project preprocessing where possible:

  * ``src.pipeline.build_preprocessor`` for the XGBoost path (handles OHE
    on ``Dispersent_Type`` and pass-through on numeric columns).
  * ``src.gpr.build_gpr_pipeline``-style standardisation for GPR (RBF /
    Matern kernels are scale-sensitive, so continuous features need
    standardising).

Per-shear-rate metric breakdown is computed by ``score_per_shear``, which
slices the predictions/labels by the ``shear_rate`` column of the test
DataFrame and returns one row per measurement shear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import Config


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, float).ravel() - np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float).ravel()
                                - np.asarray(y_pred, float).ravel())))


def score_per_shear(
    test_long: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    measurement_shears: Sequence[float],
) -> pd.DataFrame:
    """Slice the long-format predictions by shear rate and return per-shear metrics.

    Parameters
    ----------
    test_long:
        Long-format test DataFrame produced by ``data.build_augmented_split``.
        Must contain ``shear_rate`` and ``viscosity`` columns aligned with
        ``y_pred``.
    y_pred:
        Predictions for the long-format test rows, in the same order.
    measurement_shears:
        Shear rates at which the model is evaluated (the paper's three:
        1, 10, 100 s⁻¹).

    Returns
    -------
    DataFrame with one row per measurement shear and columns
    ``[shear_rate, n, r2, mae, rmse]``.
    """
    df = test_long.copy()
    df["y_pred"] = np.asarray(y_pred, float).ravel()
    rows: List[dict] = []
    for gam in measurement_shears:
        mask = np.isclose(df["shear_rate"].to_numpy(float), float(gam))
        sub = df.loc[mask]
        rows.append({
            "shear_rate": float(gam),
            "n": int(len(sub)),
            "r2": _r2(sub["viscosity"], sub["y_pred"]),
            "mae": _mae(sub["viscosity"], sub["y_pred"]),
            "rmse": _rmse(sub["viscosity"], sub["y_pred"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Learner wrappers
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Bundle of artefacts returned by every learner wrapper.

    Attributes
    ----------
    model:
        The fitted estimator (XGBRegressor / GaussianProcessRegressor /
        Pipeline wrapping one of those).
    y_pred_test:
        Predictions on the long-format test DataFrame, in the same order
        as ``test_long``.
    per_shear:
        One row per measurement shear with the standard metrics.
    overall:
        Pooled metrics across all measurement shears (a single dict).
    extra:
        Free-form bag for learner-specific diagnostics (e.g. fitted kernel
        for GPR, best Optuna params for XGB).  Every learner also stashes
        ``extra["predict_fn"]``: a callable mapping a DataFrame with the
        training feature columns to predictions in original Pa·s units,
        so downstream studies (dense curve grids, conformal calibration)
        can query any student without re-deriving its preprocessing.
    """
    model: Any
    y_pred_test: np.ndarray
    per_shear: pd.DataFrame
    overall: Dict[str, float]
    extra: Dict[str, Any]


def _make_preprocessor(X_train: pd.DataFrame, *, scale_numeric: bool) -> Any:
    """Build a ColumnTransformer that one-hot encodes categoricals.

    ``scale_numeric`` controls whether continuous features are
    standardised (set True for GPR, False for XGBoost — trees are
    scale-invariant and standardising adds nothing).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = [c for c in X_train.columns
                if pd.api.types.is_string_dtype(X_train[c])
                or pd.api.types.is_object_dtype(X_train[c])]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transformers = [("ohe", ohe, cat_cols)]
    if scale_numeric:
        transformers.append(("scale", StandardScaler(), num_cols))
    else:
        transformers.append(("pass", "passthrough", num_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _optuna_tune_xgb(
    X_train_enc: np.ndarray,
    y_fit: np.ndarray,
    weights: np.ndarray,
    *,
    n_trials: int,
    seed: int,
) -> dict:
    """Run an Optuna TPE study and return the best XGBoost hyperparameters.

    Uses a random 80/20 train/validation split (with sample weights on both
    sides) so the search is aware of the measured-vs-augmented weighting
    scheme.  The split is randomised but seeded for reproducibility.
    """
    import optuna
    from optuna.samplers import TPESampler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb  # type: ignore[import-not-found]

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_train_enc, y_fit, weights,
        test_size=0.2, random_state=seed,
    )

    def objective(trial):
        p = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "verbosity": 0,
            "random_state": seed,
        }
        m = xgb.XGBRegressor(**p)
        m.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        y_pred = m.predict(X_val)
        ss_res = float(np.sum((y_val - y_pred) ** 2))
        ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best["verbosity"] = 0
    best["random_state"] = seed
    print(f"     Optuna best  R²={study.best_value:.3f}  "
          f"params={best}")
    return best


def _monotone_constraints(preprocessor: Any) -> tuple:
    """Constraint vector pinning η non-increasing in log_gamma.

    Battery slurries are shear-thinning across the modelled 0.1–1000 s⁻¹
    range, so the student is constrained to be monotonically non-increasing
    in the encoded ``log_gamma`` column (and unconstrained elsewhere).
    Monotone in raw η implies monotone in log1p(η), so the constraint is
    valid under either target transform.
    """
    names = list(preprocessor.get_feature_names_out())
    constraints = tuple(-1 if name.endswith("log_gamma") else 0 for name in names)
    if constraints.count(-1) != 1:
        raise ValueError(
            f"Expected exactly one log_gamma column in encoded features; "
            f"found {constraints.count(-1)} among {names}"
        )
    return constraints


def fit_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    X_test: pd.DataFrame,
    test_long: pd.DataFrame,
    *,
    measurement_shears: Sequence[float],
    target_transform: str = "none",
    cfg: Optional[Config] = None,
    seed: int = 42,
) -> FitResult:
    """Fit an XGBoost regressor on the augmented long-format training set.

    When ``cfg.xgb.n_trials > 0`` an Optuna TPE study tunes the six core
    hyperparameters on a random 80/20 hold-out split (sample weights
    included).  Otherwise the project's ``xgb.fixed_params`` are used,
    falling back to sensible defaults.

    ``target_transform`` accepts ``"none"`` or ``"log1p_nonnegative"`` and
    is applied around the fit/predict so that R²/RMSE on the test set are
    reported in original Pa·s units.

    When ``cfg.physics_augmented.monotone_shear`` is True the model is
    constrained to be monotonically non-increasing in ``log_gamma``
    (shear-thinning prior; see ``_monotone_constraints``).
    """
    import xgboost as xgb  # type: ignore[import-not-found]

    preprocessor = _make_preprocessor(X_train, scale_numeric=False)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    y_fit = np.asarray(y_train, float).copy()
    if target_transform == "log1p_nonnegative":
        y_fit = np.log1p(np.maximum(y_fit, 0.0))
    elif target_transform != "none":
        raise ValueError(
            f"target_transform must be 'none' or 'log1p_nonnegative'; "
            f"got {target_transform!r}"
        )

    n_trials = int(cfg.xgb.n_trials) if cfg is not None else 0
    if n_trials > 0:
        print(f"     Optuna tuning ({n_trials} trials)…")
        params = _optuna_tune_xgb(
            X_train_enc, y_fit, np.asarray(weights_train, float),
            n_trials=n_trials, seed=seed,
        )
    else:
        params = dict(
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.01,
            n_estimators=400,
            verbosity=0,
            random_state=int(seed),
        )
        if cfg is not None:
            params.update({k: v for k, v in cfg.xgb.fixed_params.items()
                           if k not in ("n_estimators",)})
            params["n_estimators"] = int(
                cfg.xgb.fixed_params.get("n_estimators", 400)
            )

    monotone_shear = bool(
        getattr(getattr(cfg, "physics_augmented", None), "monotone_shear", False)
    )
    if monotone_shear:
        params["monotone_constraints"] = _monotone_constraints(preprocessor)

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train_enc, y_fit,
        sample_weight=np.asarray(weights_train, float),
        verbose=False,
    )

    feature_cols = list(X_train.columns)

    def _predict_frame(frame: pd.DataFrame) -> np.ndarray:
        pred = model.predict(preprocessor.transform(frame[feature_cols]))
        if target_transform == "log1p_nonnegative":
            return np.maximum(np.expm1(pred), 0.0)
        return np.asarray(pred, float)

    y_pred_test_fit = model.predict(X_test_enc)
    if target_transform == "log1p_nonnegative":
        y_pred_test = np.maximum(np.expm1(y_pred_test_fit), 0.0)
    else:
        y_pred_test = y_pred_test_fit

    per_shear = score_per_shear(test_long, y_pred_test,
                                measurement_shears=measurement_shears)
    y_true_raw = test_long["viscosity"].to_numpy(float)
    y_true_log = np.log1p(np.maximum(y_true_raw, 0.0))
    y_pred_log = np.log1p(np.maximum(np.asarray(y_pred_test, float), 0.0))
    overall = {
        "r2": _r2(y_true_raw, y_pred_test),
        "r2_log": _r2(y_true_log, y_pred_log),
        "mae": _mae(y_true_raw, y_pred_test),
        "rmse": _rmse(y_true_raw, y_pred_test),
        "n": int(len(test_long)),
    }
    return FitResult(
        model=model,
        y_pred_test=np.asarray(y_pred_test, float),
        per_shear=per_shear,
        overall=overall,
        extra={"params": params, "n_trials": n_trials,
               "predict_fn": _predict_frame},
    )


def fit_gpr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    X_test: pd.DataFrame,
    test_long: pd.DataFrame,
    *,
    measurement_shears: Sequence[float],
    target_transform: str = "none",
    cfg: Optional[Config] = None,
    seed: int = 42,
    kernel: str = "matern25",
    n_restarts: int = 3,
) -> FitResult:
    """Fit a GPR student on the augmented long-format training set.

    GPR ignores sample weights (sklearn's GaussianProcessRegressor doesn't
    expose them in a useful way).  For physics-augmented training that's
    actually fine — the augmented points are physics-faithful and treating
    them equal to measured points is a defensible default.

    ``kernel`` accepts ``"rbf"``, ``"matern25"``, ``"matern15"`` matching
    the existing ``gpr.py`` convention.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF, Matern, WhiteKernel, ConstantKernel as C
    )

    preprocessor = _make_preprocessor(X_train, scale_numeric=True)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # GPR rejects NaN; drop any rows where feature engineering produced NaN.
    nan_mask = np.isnan(X_train_enc).any(axis=1)
    if nan_mask.any():
        X_train_enc = X_train_enc[~nan_mask]
        y_train = np.asarray(y_train, float)[~nan_mask]
        weights_train = np.asarray(weights_train, float)[~nan_mask]

    y_fit = np.asarray(y_train, float).copy()
    if target_transform == "log1p_nonnegative":
        y_fit = np.log1p(np.maximum(y_fit, 0.0))
    elif target_transform != "none":
        raise ValueError(
            f"target_transform must be 'none' or 'log1p_nonnegative'; "
            f"got {target_transform!r}"
        )

    if kernel == "rbf":
        k_main = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif kernel == "matern25":
        k_main = Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 1e2))
    elif kernel == "matern15":
        k_main = Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-2, 1e2))
    else:
        raise ValueError(f"Unknown GPR kernel {kernel!r}")
    full_kernel = C(1.0, (1e-3, 1e3)) * k_main + WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-5, 1.0)
    )

    model = GaussianProcessRegressor(
        kernel=full_kernel,
        normalize_y=True,
        n_restarts_optimizer=int(n_restarts),
        random_state=int(seed),
        alpha=1e-6,
    )
    model.fit(X_train_enc, y_fit)

    feature_cols = list(X_train.columns)

    def _predict_frame(frame: pd.DataFrame) -> np.ndarray:
        pred = model.predict(preprocessor.transform(frame[feature_cols]))
        if target_transform == "log1p_nonnegative":
            return np.maximum(np.expm1(pred), 0.0)
        return np.asarray(pred, float)

    y_pred_test_fit, y_std_test_fit = model.predict(X_test_enc, return_std=True)
    if target_transform == "log1p_nonnegative":
        y_pred_test = np.maximum(np.expm1(y_pred_test_fit), 0.0)
    else:
        y_pred_test = y_pred_test_fit

    per_shear = score_per_shear(test_long, y_pred_test,
                                measurement_shears=measurement_shears)
    y_true_raw = test_long["viscosity"].to_numpy(float)
    y_true_log = np.log1p(np.maximum(y_true_raw, 0.0))
    y_pred_log = np.log1p(np.maximum(np.asarray(y_pred_test, float), 0.0))
    overall = {
        "r2": _r2(y_true_raw, y_pred_test),
        "r2_log": _r2(y_true_log, y_pred_log),
        "mae": _mae(y_true_raw, y_pred_test),
        "rmse": _rmse(y_true_raw, y_pred_test),
        "n": int(len(test_long)),
    }
    extra = {
        "kernel_fitted": str(model.kernel_),
        "log_marginal_likelihood": float(model.log_marginal_likelihood(model.kernel_.theta)),
        "y_std_test": np.asarray(y_std_test_fit, float),
        "predict_fn": _predict_frame,
    }
    return FitResult(
        model=model,
        y_pred_test=np.asarray(y_pred_test, float),
        per_shear=per_shear,
        overall=overall,
        extra=extra,
    )


def fit_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    weights_train: np.ndarray,
    X_test: pd.DataFrame,
    test_long: pd.DataFrame,
    *,
    measurement_shears: Sequence[float],
    target_transform: str = "none",
    cfg: Optional[Config] = None,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> FitResult:
    """Varying-coefficient ridge student in log space.

    The design matrix contains the encoded composition features, the
    encoded ``log_gamma`` column, and every composition × log_gamma
    product.  With the log1p target this is a power-law (Ostwald–de
    Waele) surface whose consistency and flow-index coefficients are both
    linear functions of composition — a varying-coefficient model in the
    sense of Hastie & Tibshirani (1993, JRSS-B):

        log η(x, γ̇) ≈ β₀ + βᵀx + (γ₀ + γᵀx)·log γ̇

    WHY A LINEAR STUDENT
    --------------------
    Tree ensembles predict constants outside the training hull, so under
    the strict chemistry split every genuinely new recipe falls back to
    the nearest training leaf.  Log-viscosity varies smoothly — close to
    linearly — with solids loading over the observed 69–77 wt% window
    (exponential concentration laws of the Mooney 1951 type), so a linear
    model in log space extrapolates in exactly the way trees cannot,
    while remaining queryable at any shear rate through the log_gamma
    term.

    RIDGE PENALTY SELECTION
    -----------------------
    When ``groups`` is provided (an array of chemistry IDs aligned with
    the training rows), α is selected by chemistry-level
    leave-one-out CV: each fold holds out ALL rows — measured and
    augmented — that belong to one chemistry, matching the evaluation
    protocol.  This prevents the correlated augmented rows of the same
    recipe from inflating the CV score and selecting an overly small
    penalty.  When ``groups`` is None the efficient GCV identity
    (Golub, Heath & Wahba 1979) is used as a fallback.

    Sample weights (measured vs augmented points) are honoured.
    """
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.model_selection import LeaveOneGroupOut

    preprocessor = _make_preprocessor(X_train, scale_numeric=True)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    weights = np.asarray(weights_train, float)
    nan_mask = np.isnan(X_train_enc).any(axis=1)
    if nan_mask.any():
        X_train_enc = X_train_enc[~nan_mask]
        y_train = np.asarray(y_train, float)[~nan_mask]
        weights = weights[~nan_mask]
        if groups is not None:
            groups = np.asarray(groups)[~nan_mask]

    names = list(preprocessor.get_feature_names_out())
    gamma_idx = [i for i, n in enumerate(names) if n.endswith("log_gamma")]
    if len(gamma_idx) != 1:
        raise ValueError(
            f"Expected exactly one log_gamma column; got {len(gamma_idx)} in {names}"
        )
    g = gamma_idx[0]

    def _design(X_enc: np.ndarray) -> np.ndarray:
        z_g = X_enc[:, [g]]
        others = np.delete(X_enc, g, axis=1)
        return np.hstack([X_enc, others * z_g])

    D_train = _design(X_train_enc)
    D_test = _design(X_test_enc)

    y_fit = np.asarray(y_train, float).copy()
    if target_transform == "log1p_nonnegative":
        y_fit = np.log1p(np.maximum(y_fit, 0.0))
    elif target_transform != "none":
        raise ValueError(
            f"target_transform must be 'none' or 'log1p_nonnegative'; "
            f"got {target_transform!r}"
        )

    alphas = np.logspace(-3, 3, 25)

    if groups is not None:
        # Chemistry-level LOO: each fold holds out all rows of one chemistry.
        grp = np.asarray(groups)
        logo = LeaveOneGroupOut()
        best_alpha = alphas[0]
        best_score = -np.inf
        for alpha in alphas:
            fold_scores = []
            for tr_idx, val_idx in logo.split(D_train, y_fit, grp):
                m = Ridge(alpha=alpha)
                m.fit(D_train[tr_idx], y_fit[tr_idx],
                      sample_weight=weights[tr_idx])
                y_val_pred = m.predict(D_train[val_idx])
                fold_scores.append(_r2(y_fit[val_idx], y_val_pred))
            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_alpha = alpha
        model = Ridge(alpha=best_alpha)
        model.fit(D_train, y_fit, sample_weight=weights)
        model.alpha_ = best_alpha  # expose for diagnostics
    else:
        model = RidgeCV(alphas=alphas, cv=None)
        model.fit(D_train, y_fit, sample_weight=weights)

    feature_cols = list(X_train.columns)

    def _predict_frame(frame: pd.DataFrame) -> np.ndarray:
        pred = model.predict(_design(preprocessor.transform(frame[feature_cols])))
        if target_transform == "log1p_nonnegative":
            return np.maximum(np.expm1(pred), 0.0)
        return np.asarray(pred, float)

    y_pred_test_fit = model.predict(D_test)
    if target_transform == "log1p_nonnegative":
        y_pred_test = np.maximum(np.expm1(y_pred_test_fit), 0.0)
    else:
        y_pred_test = y_pred_test_fit

    per_shear = score_per_shear(test_long, y_pred_test,
                                measurement_shears=measurement_shears)
    y_true_raw = test_long["viscosity"].to_numpy(float)
    y_true_log = np.log1p(np.maximum(y_true_raw, 0.0))
    y_pred_log = np.log1p(np.maximum(np.asarray(y_pred_test, float), 0.0))
    overall = {
        "r2": _r2(y_true_raw, y_pred_test),
        "r2_log": _r2(y_true_log, y_pred_log),
        "mae": _mae(y_true_raw, y_pred_test),
        "rmse": _rmse(y_true_raw, y_pred_test),
        "n": int(len(test_long)),
    }
    return FitResult(
        model=model,
        y_pred_test=np.asarray(y_pred_test, float),
        per_shear=per_shear,
        overall=overall,
        extra={
            "alpha": float(model.alpha_),
            "n_design_features": int(D_train.shape[1]),
            "interactions": "composition x log_gamma",
            "predict_fn": _predict_frame,
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SUPPORTED_LEARNERS = ("xgb", "gpr", "ridge")


def fit_learner(learner_name: str, **kwargs: Any) -> FitResult:
    """Dispatch ``learner_name`` to ``fit_xgb`` / ``fit_gpr`` / ``fit_ridge``.

    ``groups`` (if present in kwargs) is forwarded to ``fit_ridge`` and
    silently dropped for ``fit_xgb`` / ``fit_gpr``, which do not use it.
    """
    if learner_name == "xgb":
        kwargs.pop("groups", None)
        return fit_xgb(**kwargs)
    if learner_name == "gpr":
        kwargs.pop("groups", None)
        return fit_gpr(**kwargs)
    if learner_name == "ridge":
        return fit_ridge(**kwargs)
    raise ValueError(
        f"Unknown learner {learner_name!r}; supported: {SUPPORTED_LEARNERS}"
    )
