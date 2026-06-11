"""Chained η(10) → η(1)/η(100) prediction strategy from El-Dalahmeh et al.

The paper (Section 5, Fig. 9) describes a two-step prediction strategy:

  Step 1.  Predict η(10) from the four composition inputs.
  Step 2.  Predict η(1) and η(100) using the four composition inputs PLUS
           the predicted η(10).

The motivation is the strong empirical correlation between viscosities at
different shear rates (the paper reports r ≈ 0.92 and 0.96 between η(10) and
the other two shear rates respectively).  Letting the η(10) prediction feed
into the chained models is supposed to exploit that correlation.

LEAKAGE HAZARD AND THE THREE VARIANTS
--------------------------------------
The paper does not specify what η(10) values are fed into the chained
training data.  Three interpretations exist, each with different leakage
characteristics:

  (a) TRUE MEASURED LABELS — use the actual measured η(10) values directly
      as the chaining feature during training.  This is the strongest form
      of label-as-feature leakage and the most favourable interpretation
      for reproducing the paper's high η(1) R².  At test time, the η(10)
      model's predictions are used (no choice — labels are unknown).  The
      train-test *distribution* mismatch is small only when η(10) can be
      predicted accurately (R²≈0.81 in practice).

  (b) IN-SAMPLE PREDICTIONS — use the η(10) model's predictions on the
      same training rows it was just fit on.  For a well-fitted LightGBM
      with 500 trees these predictions are near-perfect on training data
      (functionally near-identical to case (a) on non-interpolated rows)
      and degrade at test time — creating a train-test mismatch that
      *artificially inflates* test R² relative to an honest protocol.

  (c) OUT-OF-FOLD PREDICTIONS — predict η(10) on each training row using
      a model that has never seen that row.  No train-test mismatch.
      This is the leakage-controlled chain.

Three variants are exposed via ``use_true_labels`` and ``use_oof``:

  ``fit_chained(..., use_true_labels=True)``   -- case (a): oracle training chain
  ``fit_chained(..., use_oof=False)``           -- case (b): in-sample (paper-faithful)
  ``fit_chained(..., use_oof=True)``            -- case (c): OOF (honest)

Phase 2 of the comparison reports R² for all variants and quantifies the
gap as the leakage-level estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .learners import make_learner


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChainedFitResult:
    """Container for the trained chained pipeline.

    Stores the η(10) model and the two downstream η(1) / η(100) models so
    that ``predict_chained_inference`` can reuse them on held-out data.

    Attributes
    ----------
    learner_name:
        ``'lightgbm'`` / ``'ols'`` / ``'ridge'`` — the same learner is used
        for every step of the chain, matching the paper's setup.
    eta10_model:
        Fitted learner that maps composition → η(10).
    eta1_model:
        Fitted learner that maps (composition, predicted η(10)) → η(1).
    eta100_model:
        Fitted learner that maps (composition, predicted η(10)) → η(100).
    eta10_target_col, eta1_target_col, eta100_target_col:
        Target column names; carried for downstream report-building.
    feature_cols:
        Encoded composition column names used for all three models (η(1) and
        η(100) get one additional column appended at predict time).
    """
    learner_name: str
    eta10_model: Any
    eta1_model: Any
    eta100_model: Any
    eta10_target_col: str
    eta1_target_col: str
    eta100_target_col: str
    feature_cols: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_eta10(X: np.ndarray, eta10: np.ndarray) -> np.ndarray:
    """Concatenate predicted (or measured) η(10) as an extra feature column."""
    eta10 = np.asarray(eta10, float).reshape(-1, 1)
    return np.concatenate([np.asarray(X, float), eta10], axis=1)


def _oof_predict(
    learner_name: str,
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    learner_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions for the η(10) model on (source, target_train).

    The TL pipeline always pre-trains on the full source batch, so leakage
    only matters within the target-train rows that feed the chained η(1) /
    η(100) models.  Implementation:

      For each of K folds of the target_train rows:
        - Hold out fold k as a "calibration" set.
        - Fit the learner on (full source batch) + (target_train minus fold k).
        - Predict on fold k.
      Concatenate the per-fold predictions into a single OOF vector.

    Source-batch rows always see their own label in the η(10) training set,
    by design — the no-leakage requirement is only on the target-train rows
    because those are what the chained models use as input features at fit
    time.  Source rows are predicted by the final full-data model after the
    OOF loop completes.

    Returns
    -------
    (eta10_oof_target, eta10_full_source)
        Out-of-fold predictions for ``X_target_train`` and full-data
        predictions for ``X_source`` respectively, suitable for
        concatenation into the chained-model feature matrix.
    """
    learner_kwargs = dict(learner_kwargs or {})
    rng = np.random.default_rng(seed)

    n_target = len(X_target_train)
    if n_target < n_splits:
        # Too few target rows for the requested K — fall back to leave-one-out.
        n_splits = max(2, n_target)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    eta10_oof_target = np.zeros(n_target, dtype=float)
    for train_idx, test_idx in kf.split(X_target_train):
        learner = make_learner(learner_name, seed=seed, **learner_kwargs)
        learner.fit(
            X_source,
            y_source,
            X_target_train[train_idx],
            y_target_train[train_idx],
        )
        eta10_oof_target[test_idx] = learner.predict(X_target_train[test_idx])

    # Final η(10) model fit on everything; used to predict source-batch rows
    # (where leakage doesn't matter for the chained pipeline) and to predict
    # any held-out test rows downstream.
    full_learner = make_learner(learner_name, seed=seed, **learner_kwargs)
    full_learner.fit(X_source, y_source, X_target_train, y_target_train)
    eta10_full_source = full_learner.predict(X_source)
    return eta10_oof_target, eta10_full_source


# ---------------------------------------------------------------------------
# Fit / predict
# ---------------------------------------------------------------------------


def fit_chained(
    learner_name: str,
    X_source: np.ndarray,
    y_source: Dict[str, np.ndarray],
    X_target_train: np.ndarray,
    y_target_train: Dict[str, np.ndarray],
    *,
    eta10_target: str,
    eta1_target: str,
    eta100_target: str,
    feature_cols: List[str],
    use_true_labels: bool = False,
    use_oof: bool = False,
    oof_n_splits: int = 5,
    seed: int = 42,
    learner_kwargs: Optional[Dict[str, Any]] = None,
) -> ChainedFitResult:
    """Fit the chained η(10) → η(1)/η(100) pipeline.

    Parameters
    ----------
    learner_name:
        Learner key used for every step of the chain (uniform across the
        chain matches the paper's description).
    X_source, X_target_train:
        Encoded composition feature matrices for the source batch and the
        training portion of the target batch respectively.
    y_source, y_target_train:
        Dicts keyed by viscosity-target column name; each entry is a 1-D
        array aligned with the corresponding ``X``.
    eta10_target, eta1_target, eta100_target:
        Column names for the three viscosity targets.  The η(10) target is
        used as both an input (to chain into η(1) / η(100)) and an output
        (to evaluate on the test set later).
    feature_cols:
        Column names of the composition features.  Stored for downstream
        reporting; not used at fit time except for the result object.
    use_true_labels:
        When True, use the actual measured η(10) values (``y_source[eta10]``
        and ``y_target_train[eta10]``) directly as the chaining feature at
        training time — the most aggressive/faithful interpretation of the
        paper's method.  At test time the η(10) model's predictions are
        always used (no measured labels available).  Takes precedence over
        ``use_oof`` when both are True.
    use_oof:
        When False (default), use the η(10) model's in-sample predictions on
        the training data — the paper-faithful chain.
        When True, use K-fold out-of-fold predictions on the target_train
        rows — the leakage-controlled chain.
        Ignored when ``use_true_labels=True``.
    oof_n_splits:
        Number of folds when ``use_oof`` is True.
    seed:
        Reproducibility seed forwarded to every learner.
    learner_kwargs:
        Extra arguments forwarded to ``make_learner``.
    """
    learner_kwargs = dict(learner_kwargs or {})

    y10_src = np.asarray(y_source[eta10_target], float).ravel()
    y10_tgt = np.asarray(y_target_train[eta10_target], float).ravel()
    y1_src = np.asarray(y_source[eta1_target], float).ravel()
    y1_tgt = np.asarray(y_target_train[eta1_target], float).ravel()
    y100_src = np.asarray(y_source[eta100_target], float).ravel()
    y100_tgt = np.asarray(y_target_train[eta100_target], float).ravel()

    # Step 1: η(10) model.
    eta10_model = make_learner(learner_name, seed=seed, **learner_kwargs)
    eta10_model.fit(X_source, y10_src, X_target_train, y10_tgt)

    # Step 2: build the augmented feature matrices for the downstream models.
    if use_true_labels:
        # Most faithful reproduction: use the actual measured η(10) values
        # as the chaining feature during training.  This eliminates all
        # train-test mismatch at training time and is the most generous
        # interpretation of the paper's chaining description.
        # At test time (predict_chained) the η(10) model's predictions are
        # still used — there is no alternative.
        eta10_for_source = y10_src
        eta10_for_target = y10_tgt
    elif use_oof:
        eta10_oof_target, eta10_full_source = _oof_predict(
            learner_name,
            np.asarray(X_source, float),
            y10_src,
            np.asarray(X_target_train, float),
            y10_tgt,
            n_splits=oof_n_splits,
            seed=seed,
            learner_kwargs=learner_kwargs,
        )
        eta10_for_source = eta10_full_source
        eta10_for_target = eta10_oof_target
    else:
        # Paper-faithful: in-sample predictions from the freshly-trained
        # η(10) model are used as the η(10) feature for the chained models.
        eta10_for_source = eta10_model.predict(X_source)
        eta10_for_target = eta10_model.predict(X_target_train)

    X_source_aug = _append_eta10(X_source, eta10_for_source)
    X_target_aug = _append_eta10(X_target_train, eta10_for_target)

    # Step 3: η(1) and η(100) models on the augmented features.
    eta1_model = make_learner(learner_name, seed=seed, **learner_kwargs)
    eta1_model.fit(X_source_aug, y1_src, X_target_aug, y1_tgt)

    eta100_model = make_learner(learner_name, seed=seed, **learner_kwargs)
    eta100_model.fit(X_source_aug, y100_src, X_target_aug, y100_tgt)

    return ChainedFitResult(
        learner_name=learner_name,
        eta10_model=eta10_model,
        eta1_model=eta1_model,
        eta100_model=eta100_model,
        eta10_target_col=eta10_target,
        eta1_target_col=eta1_target,
        eta100_target_col=eta100_target,
        feature_cols=list(feature_cols),
    )


def predict_chained(
    fit_result: ChainedFitResult,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Predict η(1), η(10), η(100) for held-out rows.

    At test time the chained pipeline is unambiguous — there is no
    train-test mismatch question because the test rows have never been
    seen at training time regardless of variant.

    Returns
    -------
    Dict keyed by target column name with 1-D prediction arrays.
    """
    X = np.asarray(X, float)
    eta10 = fit_result.eta10_model.predict(X)
    X_aug = _append_eta10(X, eta10)
    eta1 = fit_result.eta1_model.predict(X_aug)
    eta100 = fit_result.eta100_model.predict(X_aug)
    return {
        fit_result.eta1_target_col: np.asarray(eta1, float),
        fit_result.eta10_target_col: np.asarray(eta10, float),
        fit_result.eta100_target_col: np.asarray(eta100, float),
    }


# ---------------------------------------------------------------------------
# Direct (no-chain) variant — the honest baseline for Phase 2
# ---------------------------------------------------------------------------


@dataclass
class DirectFitResult:
    """Three independent learners, one per viscosity target — no chaining."""
    learner_name: str
    models: Dict[str, Any] = field(default_factory=dict)
    feature_cols: List[str] = field(default_factory=list)


def fit_direct(
    learner_name: str,
    X_source: np.ndarray,
    y_source: Dict[str, np.ndarray],
    X_target_train: np.ndarray,
    y_target_train: Dict[str, np.ndarray],
    *,
    target_cols: List[str],
    feature_cols: List[str],
    seed: int = 42,
    learner_kwargs: Optional[Dict[str, Any]] = None,
) -> DirectFitResult:
    """Fit one learner per viscosity target, all from composition only.

    The "no chaining" variant referenced in Phase 2.  Removes the η(10)
    crutch entirely so that R²(1) and R²(100) reflect what the four
    composition inputs alone can predict.
    """
    learner_kwargs = dict(learner_kwargs or {})
    models: Dict[str, Any] = {}
    for tgt in target_cols:
        ys = np.asarray(y_source[tgt], float).ravel()
        yt = np.asarray(y_target_train[tgt], float).ravel()
        m = make_learner(learner_name, seed=seed, **learner_kwargs)
        m.fit(X_source, ys, X_target_train, yt)
        models[tgt] = m
    return DirectFitResult(
        learner_name=learner_name,
        models=models,
        feature_cols=list(feature_cols),
    )


def predict_direct(
    fit_result: DirectFitResult,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    X = np.asarray(X, float)
    return {
        tgt: np.asarray(model.predict(X), float)
        for tgt, model in fit_result.models.items()
    }
