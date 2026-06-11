"""Learners replicated from El-Dalahmeh et al. Section 6.

Three learner families are provided:

  * ``LightGBMTL``  -- gradient-boosted trees with ``init_model``
                       continuation training.  This is the paper's headline
                       "transfer learning" mechanism: 500 trees fit on
                       Batch_1 at lr=0.10, then up to 100 additional trees
                       appended at lr=0.05 using Batch_2 data.  Section 7.3
                       and Table 3 of the paper.
  * ``LinearTL``    -- ordinary least squares baseline (Section 6.3).
                       "Transfer" here is just retraining on the union of
                       source and target batches, since OLS has no notion of
                       continuation training.
  * ``RidgeTL``     -- ridge baseline with α selected by inner k-fold CV on
                       the source batch alone (matches the paper's
                       description of selecting α "via inner k-fold CV on
                       the training partition").

All three implement the same minimal interface so the chained-prediction
strategy and the conformal wrapper can drive them uniformly:

    learner.fit(X_source, y_source, X_target_train=None, y_target_train=None)
    learner.predict(X) -> np.ndarray

When ``X_target_train`` and ``y_target_train`` are provided to
``LightGBMTL.fit``, the booster is fit on the source data first and then
continuation-trained on the target data.  When they are omitted, only the
source-batch fit is performed (the "no-transfer" reference reported by the
paper as its negative control).

LightGBM is imported lazily inside ``LightGBMTL.fit`` so the module remains
importable on machines where ``lightgbm`` is not installed.  sklearn imports
happen at module import time because sklearn is already a hard dependency
elsewhere in the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


# ---------------------------------------------------------------------------
# LightGBM with continuation training
# ---------------------------------------------------------------------------


@dataclass
class LightGBMConfig:
    """Hyperparameters mirroring El-Dalahmeh et al. Table 3.

    Attributes
    ----------
    n_base_trees:
        Number of boosting rounds for the source-batch (Batch_1) fit.
        The paper specifies 500.
    n_finetune_trees:
        Maximum additional rounds appended during continuation training on
        the target batch (Batch_2).  The paper specifies "up to 100".
    learning_rate_base:
        Learning rate for the source-batch fit.  Paper: 0.10.
    learning_rate_finetune:
        Learning rate for the continuation rounds.  Paper: 0.05.
    num_leaves, max_depth, min_data_in_leaf, feature_fraction,
    bagging_fraction, bagging_freq, lambda_l2:
        Standard LightGBM regulariser knobs.  The paper does not report
        exact values, so LightGBM library defaults are used (chosen to be
        conservative for small datasets).
    seed:
        Reproducibility seed forwarded to LightGBM.
    """
    n_base_trees: int = 500
    n_finetune_trees: int = 100
    learning_rate_base: float = 0.10
    learning_rate_finetune: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_data_in_leaf: int = 5
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 0.0
    seed: int = 42


@dataclass
class LightGBMTL:
    """LightGBM regressor with init_model continuation transfer learning."""
    config: LightGBMConfig = field(default_factory=LightGBMConfig)
    _booster: Any = field(default=None, init=False, repr=False)
    _n_features: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------ fit

    def _base_params(self, learning_rate: float) -> Dict[str, Any]:
        c = self.config
        return {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": learning_rate,
            "num_leaves": c.num_leaves,
            "max_depth": c.max_depth,
            "min_data_in_leaf": c.min_data_in_leaf,
            "feature_fraction": c.feature_fraction,
            "bagging_fraction": c.bagging_fraction,
            "bagging_freq": c.bagging_freq,
            "lambda_l2": c.lambda_l2,
            "verbose": -1,
            "seed": c.seed,
            "deterministic": True,
            "force_row_wise": True,
        }

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target_train: Optional[np.ndarray] = None,
        y_target_train: Optional[np.ndarray] = None,
    ) -> "LightGBMTL":
        """Fit on the source batch, then continuation-train on the target.

        When ``X_target_train`` is None, only the source-batch fit is
        performed — useful for the paper's no-transfer reference.

        Continuation training reuses the source-batch booster as the
        ``init_model`` for a second ``lgb.train`` call with the target-batch
        data and the lower fine-tuning learning rate.  This matches the
        "update mode" described in Section 7.3 of the paper.
        """
        # Lazy import keeps the module importable without lightgbm available.
        import lightgbm as lgb  # type: ignore[import-not-found]

        X_source = np.asarray(X_source, dtype=float)
        y_source = np.asarray(y_source, dtype=float).ravel()
        self._n_features = X_source.shape[1]

        base_params = self._base_params(self.config.learning_rate_base)
        train_set = lgb.Dataset(X_source, label=y_source)
        booster = lgb.train(
            base_params,
            train_set,
            num_boost_round=self.config.n_base_trees,
        )

        if (X_target_train is not None and y_target_train is not None
                and len(X_target_train) > 0):
            ft_params = self._base_params(self.config.learning_rate_finetune)
            ft_set = lgb.Dataset(
                np.asarray(X_target_train, dtype=float),
                label=np.asarray(y_target_train, dtype=float).ravel(),
            )
            booster = lgb.train(
                ft_params,
                ft_set,
                num_boost_round=self.config.n_finetune_trees,
                init_model=booster,
                keep_training_booster=False,
            )

        self._booster = booster
        return self

    # -------------------------------------------------------------- predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("LightGBMTL.predict called before fit")
        X = np.asarray(X, dtype=float)
        if len(X) == 0:
            return np.empty(0, dtype=float)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {X.shape[1]}"
            )
        return np.asarray(self._booster.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# Linear baselines (Section 6.3)
# ---------------------------------------------------------------------------


@dataclass
class LinearTL:
    """OLS baseline.  "Transfer" = refit on source ∪ target_train."""
    _model: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target_train: Optional[np.ndarray] = None,
        y_target_train: Optional[np.ndarray] = None,
    ) -> "LinearTL":
        if X_target_train is not None and y_target_train is not None:
            X = np.vstack([np.asarray(X_source, float),
                           np.asarray(X_target_train, float)])
            y = np.concatenate([np.asarray(y_source, float).ravel(),
                                np.asarray(y_target_train, float).ravel()])
        else:
            X = np.asarray(X_source, float)
            y = np.asarray(y_source, float).ravel()
        self._model = LinearRegression().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LinearTL.predict called before fit")
        X = np.asarray(X, float)
        if len(X) == 0:
            return np.empty(0, dtype=float)
        return np.asarray(self._model.predict(X), dtype=float)


@dataclass
class RidgeTL:
    """Ridge baseline with inner-CV α selection on the source batch.

    α is selected by ``RidgeCV`` over a log-spaced grid using leave-one-out
    cross-validation on the source batch only — matching the paper's
    description in Section 6.3.  When ``X_target_train`` is supplied, the
    selected α is reused and the model is refit on the union of source and
    target_train rows.
    """
    alphas: Sequence[float] = field(
        default_factory=lambda: tuple(np.logspace(-3, 3, 25))
    )
    _alpha: float = field(default=1.0, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    def fit(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target_train: Optional[np.ndarray] = None,
        y_target_train: Optional[np.ndarray] = None,
    ) -> "RidgeTL":
        Xs = np.asarray(X_source, float)
        ys = np.asarray(y_source, float).ravel()
        # cv=None uses efficient leave-one-out inside RidgeCV.
        cv_model = RidgeCV(alphas=tuple(self.alphas), cv=None).fit(Xs, ys)
        self._alpha = float(cv_model.alpha_)

        if X_target_train is not None and y_target_train is not None:
            Xt = np.asarray(X_target_train, float)
            yt = np.asarray(y_target_train, float).ravel()
            X = np.vstack([Xs, Xt])
            y = np.concatenate([ys, yt])
        else:
            X = Xs
            y = ys
        self._model = Ridge(alpha=self._alpha).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("RidgeTL.predict called before fit")
        X = np.asarray(X, float)
        if len(X) == 0:
            return np.empty(0, dtype=float)
        return np.asarray(self._model.predict(X), dtype=float)

    @property
    def alpha(self) -> float:
        return self._alpha


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def make_learner(name: str, *, seed: int = 42, **kwargs: Any) -> Any:
    """Construct a learner by string name.

    Parameters
    ----------
    name:
        One of ``'lightgbm'``, ``'ols'``, ``'ridge'``.
    seed:
        Reproducibility seed forwarded to learners that consume one.
    kwargs:
        Forwarded to ``LightGBMConfig`` for ``'lightgbm'`` (ignored otherwise).
    """
    name_lc = name.lower()
    if name_lc == "lightgbm":
        cfg = LightGBMConfig(seed=seed, **kwargs)
        return LightGBMTL(config=cfg)
    if name_lc in ("ols", "linear", "linear_regression"):
        return LinearTL()
    if name_lc == "ridge":
        return RidgeTL()
    raise ValueError(
        f"Unknown TL learner {name!r}.  Valid: 'lightgbm', 'ols', 'ridge'."
    )


_VALID_LEARNERS = ("lightgbm", "ols", "ridge")
