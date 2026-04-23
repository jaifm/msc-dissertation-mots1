"""XGBoost hyperparameter tuning via Optuna and held-out evaluation.

OVERVIEW
--------
This module provides the single entry point used by all training paths in the
project: ``tune_and_evaluate_xgb``.  It takes a train/test split, handles
feature engineering and preprocessing, searches for good XGBoost
hyperparameters using Optuna, trains a final model with those parameters, and
evaluates it on the held-out test set.

TUNING STRATEGY
---------------
Optuna runs a Bayesian optimisation loop (Tree-structured Parzen Estimator,
TPE) over the XGBoost hyperparameter space for ``cfg.xgb.n_trials`` trials.
Each trial trains on 80% of the training set and evaluates R² on the remaining
20% as a held-out validation split.

Why Optuna over sklearn's GridSearchCV or RandomizedSearchCV?
  - TPE focuses trials on promising regions of the search space rather than
    sampling uniformly, which is more efficient with small trial budgets.
  - The 80/20 internal split avoids cross-validation overhead, which matters
    when running 5 models × multiple targets × potentially many experiments.
  - The final model after tuning is retrained on the FULL training set (not
    just the 80% fold), so no data is wasted on the validation split.

SEARCH SPACE
------------
The hyperparameters searched per trial are:
  max_depth        [3, 10]        -- tree depth; deeper = more expressive but slower to train
  learning_rate    [0.001, 0.3]   -- step size (log scale); lower needs more trees to converge
  subsample        [0.5, 1.0]     -- row sampling fraction per tree; reduces variance
  colsample_bytree [0.5, 1.0]     -- feature sampling fraction per tree; reduces correlation
  lambda (L2)      [0.001, 10]    -- weight decay (log scale); shrinks leaf values
  alpha  (L1)      [0.001, 10]    -- sparsity regularisation (log scale)
  min_child_weight [1, 10]        -- minimum sum of instance weights in a leaf; critical for
                                     small datasets — higher values prevent the model from
                                     fitting splits that are driven by only 1–2 rows
  gamma            [0.0, 5.0]     -- minimum loss reduction required to make a split; acts as
                                     a tree-complexity penalty that prunes splits with marginal
                                     benefit, complementing L2/L1 weight regularisation

EARLY STOPPING
--------------
Each Optuna trial uses ``_N_ESTIMATORS_SEARCH`` trees with early stopping
(``_EARLY_STOPPING_ROUNDS`` rounds of no validation-R² improvement).  This
serves two purposes:

  1. **Efficiency**: trials that plateau early stop rather than wasting time
     training trees that no longer improve the held-out score.
  2. **Best-iteration transfer**: the trial stores its best_iteration as a
     user attribute.  The winning trial's best_iteration is used directly as
     n_estimators for the final model, rather than the fixed config value.
     This is more principled than a hand-picked constant because it is derived
     from the actual learning curve of the best hyperparameter combination.

The final model is retrained on the FULL training set (no held-out validation
fold) using the early-stopped iteration count.  Because training on ~25% more
data tends to delay convergence slightly, the iteration count is multiplied by
``_FINAL_N_EST_SCALE`` (default 1.1) before use and capped at
``_FINAL_N_EST_CAP``.  If early stopping did not fire for the best trial,
``cfg.xgb.n_estimators`` is used as a fallback.

TARGET TRANSFORM
----------------
Rheology parameters (η_0, λ, K etc.) are positive and right-skewed.  For
some targets, the distribution has a long tail that causes XGBoost to focus
its splits on the high-value end and perform poorly on the majority of rows.
``log1p_nonnegative`` mode applies log1p(y + shift) before training and
inverts it after prediction.  The shift ensures y + shift > 0 for any
target that can take negative values (e.g. η_inf for some fits).

The transform is derived entirely from the training set to avoid leakage.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler

from .config import Config
from .features import engineer_composition_features
from .pipeline import make_preprocessor, r2_mse_rmse

# Silence Optuna's per-trial INFO spam; the calling sweep loop handles its own
# progress output and double-printing every trial wastes notebook space.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Early-stopping tuning constants
# ---------------------------------------------------------------------------

# Upper bound on n_estimators used inside each Optuna trial.  Trials stop
# early via _EARLY_STOPPING_ROUNDS so most will never reach this cap.
# Setting it to 300 (vs the old fixed 100) gives low-learning-rate configs
# enough trees to actually converge, while early stopping prevents waste.
_N_ESTIMATORS_SEARCH: int = 300

# Number of consecutive validation-R² rounds with no improvement before a
# trial is cut short.  20 is conservative enough that genuine plateaus are
# detected without cutting off configs that improve slowly.
_EARLY_STOPPING_ROUNDS: int = 20

# Scale factor applied to the best trial's early-stopped iteration count when
# setting n_estimators for the final model.  The final model trains on ~25%
# more data (full training set vs 80% fold), so it typically needs slightly
# more trees to reach the same loss level.  1.1 is a small, principled
# correction; set to 1.0 to disable scaling.
_FINAL_N_EST_SCALE: float = 1.1

# Hard cap on the final model's n_estimators after scaling.  Prevents
# pathological cases where best_iteration is very large from producing an
# impractically slow final fit.
_FINAL_N_EST_CAP: int = 500


def _make_xgb_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    objective_fn: str = "reg:squarederror",
    objective_params: dict | None = None,
):
    """Return an Optuna objective function that scores one hyperparameter trial.

    The objective uses a fixed 80/20 chronological split of the training data
    rather than cross-validation.  Using a single split is intentional:
    - With ~100–150 training rows, 5-fold CV reduces each fold's training set
      to ~80 rows, which is too small for reliable tree fitting.
    - A single split is reproducible (no random fold assignment) and fast.

    The split is positional (first 80% / last 20%), not random, so the
    validation set behaviour is deterministic across trials within a study.

    Parameters
    ----------
    X_train:
        Preprocessed feature matrix for the full training set.
    y_train:
        Target values (possibly log-transformed) for the full training set.
    seed:
        Random seed passed to XGBoost for reproducibility.

    Returns
    -------
    Callable
        Optuna objective that accepts a ``Trial`` and returns a float R².
        After each trial completes, ``trial.user_attrs['best_iteration']``
        holds the early-stopped tree count, which the caller uses to set
        n_estimators for the final model.
    """
    n = len(X_train)
    split = int(0.8 * n)
    X_fit, X_val = X_train[:split], X_train[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    extra = objective_params or {}

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            # min_child_weight: minimum sum of instance weights required in a
            # leaf node before a split is accepted.  With ~100–150 training
            # rows, the default of 1 allows splits on single rows, which
            # overfit heavily.  Searching [1, 10] lets Optuna discover how
            # much leaf-size regularisation this dataset needs.
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # gamma: minimum loss-reduction gain for a split to be accepted.
            # Complementary to L2/L1 regularisation — it prunes splits that
            # reduce training loss only marginally, acting as an explicit
            # tree-complexity penalty.
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": objective_fn,
            **extra,
            "random_state": seed,
            "n_estimators": _N_ESTIMATORS_SEARCH,
            # early_stopping_rounds is set on the constructor (XGBoost ≥ 1.7
            # sklearn API); eval_set passed to fit() activates it.
            "early_stopping_rounds": _EARLY_STOPPING_ROUNDS,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        # Store the early-stopped iteration so the best trial's count can be
        # used as n_estimators in the final model (see tune_and_evaluate_xgb).
        trial.set_user_attr("best_iteration", int(model.best_iteration))
        r2, _, _ = r2_mse_rmse(y_val, model.predict(X_val))
        return r2

    return objective


def _apply_target_transform(y: np.ndarray, mode: str) -> Tuple[np.ndarray, float]:
    """Apply an optional monotonic transform to the target variable.

    Transforming the target can improve XGBoost's fit when the target
    distribution is strongly right-skewed, because gradient boosting
    minimises squared error and large residuals on extreme values dominate
    the gradient signal.

    Parameters
    ----------
    y:
        Raw target values from the training set.
    mode:
        ``'none'`` -- no transform, return y unchanged.
        ``'log1p_nonnegative'`` -- shift y so min > 0, then apply log1p.
            The shift is computed from training data only.

    Returns
    -------
    (transformed_y, shift)
        ``shift`` must be passed to ``_invert_target_transform`` at prediction
        time to recover predictions on the original scale.
    """
    if mode == "none":
        return y, 0.0
    if mode == "log1p_nonnegative":
        min_y = float(np.min(y))
        # If the minimum is already > -1 the shift is zero and log1p(y) is
        # safe.  Otherwise shift by |min| + 1.001 so the smallest value maps
        # to log1p(0.001) > 0.
        shift = abs(min_y) + 1.001 if min_y <= -1.0 else 0.0
        return np.log1p(y + shift), shift
    raise ValueError(f"Unknown target_transform mode: {mode}")


def _invert_target_transform(y_pred: np.ndarray, mode: str, shift: float) -> np.ndarray:
    """Invert the transform applied by ``_apply_target_transform``.

    This must be called on model predictions before computing metrics or
    storing results, so that all reported numbers are on the original physical
    scale (e.g. Pa·s for viscosity parameters).
    """
    if mode == "none":
        return y_pred
    if mode == "log1p_nonnegative":
        # expm1(x) - shift is the exact inverse of log1p(y + shift).
        return np.expm1(y_pred) - shift
    raise ValueError(mode)


def _select_feature_cols(df: pd.DataFrame, cfg: Config) -> list:
    """Return the subset of ``cfg.data.composition_cols`` present in ``df``.

    The intersection is used rather than the full list so that this function
    is safe to call on DataFrames that have already been column-filtered (e.g.
    a leave-one-out fold that had a column dropped).
    """
    return [c for c in cfg.data.composition_cols if c in df.columns]


def tune_and_evaluate_xgb(
    physics_train: pd.DataFrame,
    physics_test: pd.DataFrame,
    target_col: str,
    cfg: Config,
    *,
    log_numeric_features: bool | None = None,
) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    """Tune XGBoost hyperparameters with Optuna, then evaluate on the test set.

    This is the core training function used by both the baseline sweep and the
    log-numeric variant comparison.  It performs the following steps in order:

      1. Select composition feature columns from the physics DataFrames.
      2. Optionally apply feature engineering (``cfg.features.use_engineered``).
      3. Build a preprocessing pipeline fitted on training data only.
      4. Optionally transform the target (``cfg.xgb.target_transform``).
      5. Run Optuna to find the best hyperparameters.
      6. Retrain on the full training set with the best parameters.
      7. Evaluate on the held-out test set; invert any target transform.
      8. Return a result dict, a predictions DataFrame, and train predictions.

    Parameters
    ----------
    physics_train, physics_test:
        DataFrames with fitted physics parameters appended (from
        ``build_physics_dataset``).  Both must contain the composition columns
        listed in ``cfg.data.composition_cols``.
    target_col:
        Name of the column to predict (e.g. ``'eta_0'``, ``'n_flow'``).
    cfg:
        Full experiment configuration.
    log_numeric_features:
        When set, overrides ``cfg.features.log_numeric`` for this call only.
        Used by the sweep section to compare variants without mutating the
        config object.

    Returns
    -------
    (result_dict, predictions_df, train_predictions_array)
        ``result_dict`` contains R², MSE, RMSE for train and test, plus
        the number of Optuna trials and the best hyperparameters.
        ``predictions_df`` has columns [y_true, y_pred] indexed to the test set.
        ``train_predictions_array`` is the model's predictions on the training
        set (used for overfitting diagnostics).
    """
    if log_numeric_features is None:
        log_numeric_features = cfg.features.log_numeric

    # --- Step 1-2: feature selection and engineering -------------------------
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

    # --- Step 3: preprocessing -----------------------------------------------
    # Fit the preprocessor on training data only; transform both splits.
    # This ensures that OHE category levels and imputation statistics are
    # derived from the training distribution, not the test distribution.
    preprocessor = make_preprocessor(X_train, log_numeric_features=log_numeric_features)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- Step 4: target transform --------------------------------------------
    y_train_model, shift = _apply_target_transform(y_train, cfg.xgb.target_transform)

    # --- Step 5: Optuna hyperparameter search --------------------------------
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=cfg.seed))
    study.optimize(
        _make_xgb_objective(
            X_train_proc, y_train_model, cfg.seed,
            objective_fn=cfg.xgb.objective,
            objective_params=cfg.xgb.objective_params,
        ),
        n_trials=cfg.xgb.n_trials,
        show_progress_bar=False,
    )

    # --- Step 6: retrain on full training set --------------------------------
    # Derive n_estimators from the early-stopped iteration count of the best
    # Optuna trial rather than using a fixed config value.  The iteration count
    # is scaled up by _FINAL_N_EST_SCALE to account for the ~25% larger
    # training set (full set vs the 80% fold used during search), then capped
    # at _FINAL_N_EST_CAP.  If early stopping never fired (edge case), fall
    # back to cfg.xgb.n_estimators.
    raw_best_iter = study.best_trial.user_attrs.get("best_iteration", cfg.xgb.n_estimators)
    # Floor at 10 to guard against early stopping firing on the first tree
    # (can happen when a trial's hyperparameters are degenerate).
    final_n_estimators = max(10, min(int(raw_best_iter * _FINAL_N_EST_SCALE), _FINAL_N_EST_CAP))

    best_params = dict(study.best_params)
    best_params.update(
        {
            "objective": cfg.xgb.objective,
            **cfg.xgb.objective_params,
            "n_estimators": final_n_estimators,
            "random_state": cfg.seed,
            "verbosity": 0,
        }
    )
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train_proc, y_train_model)

    # --- Step 7: evaluate and invert transform --------------------------------
    y_train_pred_model = final_model.predict(X_train_proc)
    y_test_pred_model = final_model.predict(X_test_proc)

    y_train_pred = _invert_target_transform(y_train_pred_model, cfg.xgb.target_transform, shift)
    y_test_pred = _invert_target_transform(y_test_pred_model, cfg.xgb.target_transform, shift)

    train_r2, train_mse, train_rmse = r2_mse_rmse(y_train, y_train_pred)
    test_r2, test_mse, test_rmse = r2_mse_rmse(y_test, y_test_pred)

    # --- Step 8: package results ---------------------------------------------
    result = {
        "target": target_col,
        "train_r2": train_r2,
        "train_mse": train_mse,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "n_trials": cfg.xgb.n_trials,
        "best_params": best_params,
        # n_estimators actually used for the final fit (early-stopping derived).
        # Stored separately so it can be inspected alongside the Optuna params.
        "final_n_estimators": final_n_estimators,
        "log_numeric_features": log_numeric_features,
        "target_transform": cfg.xgb.target_transform,
    }

    predictions_df = pd.DataFrame(
        {"y_true": y_test, "y_pred": y_test_pred, "target": target_col}
    )

    return result, predictions_df, y_train_pred
