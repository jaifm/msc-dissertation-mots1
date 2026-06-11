"""Split-conformal prediction intervals (El-Dalahmeh et al. Section 8.1).

The paper reports "distribution-free prediction intervals constructed using
split-conformal prediction on the target batch" and specifies the protocol:

  For each k-fold within the target batch:
    - Split the training portion of the fold into a proper-training set and
      a calibration set.
    - Fit a base predictor on the proper-training set.
    - Compute absolute residuals on the calibration set.
    - Take the (1−α) empirical quantile of those residuals as q.
    - Form the two-sided interval [ŷ − q, ŷ + q] around each test prediction.

  Report empirical coverage (fraction of test points whose true value falls
  inside the interval) and mean interval width on the held-out test rows.

This module supplies:

  * ``split_conformal_quantile``  -- the residual-quantile primitive
  * ``conformal_intervals``       -- two-sided interval helper
  * ``conformal_coverage``        -- empirical coverage scorer
  * ``ConformalReport``           -- summary statistics

The base predictor is treated as a black box conforming to the ``fit`` /
``predict`` contract used by the learners in this subpackage, so the same
function can wrap LightGBM, OLS, Ridge, or any other learner that exposes
those two methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConformalReport:
    """Summary of conformal-interval performance at one α.

    Attributes
    ----------
    alpha:
        Nominal miscoverage level (e.g. 0.10 for a 90% interval).
    coverage:
        Empirical fraction of test rows whose true target falls inside the
        interval.  A well-calibrated split-conformal procedure yields
        coverage ≈ 1−α in expectation over fresh calibration sets.
    mean_width:
        Average width of the two-sided interval across test rows.  Units
        match the target (here Pa·s).
    n_test:
        Number of test rows the report aggregates over.
    """
    alpha: float
    coverage: float
    mean_width: float
    n_test: int


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def split_conformal_quantile(
    y_true_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    alpha: float,
) -> float:
    """Return the (1−α) quantile of absolute residuals on a calibration set.

    The conformal correction multiplies the empirical quantile by
    ``(n+1)/n`` so that the resulting prediction set has marginal coverage
    at least 1−α in finite samples (Lei et al. 2018 / Romano et al. 2019).
    """
    y_true_calib = np.asarray(y_true_calib, float).ravel()
    y_pred_calib = np.asarray(y_pred_calib, float).ravel()
    if y_true_calib.shape != y_pred_calib.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true_calib.shape} vs "
            f"y_pred {y_pred_calib.shape}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    n = len(y_true_calib)
    if n == 0:
        raise ValueError("Empty calibration set")
    residuals = np.abs(y_true_calib - y_pred_calib)
    # Conformal finite-sample correction: take the ceil((n+1)(1-α))/n quantile.
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(rank, n)  # cap so we never index past the sample
    return float(np.sort(residuals)[rank - 1])


def conformal_intervals(
    y_pred: np.ndarray,
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (lower, upper) bounds of the symmetric conformal interval."""
    y_pred = np.asarray(y_pred, float).ravel()
    return y_pred - quantile, y_pred + quantile


def conformal_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[float, float, int]:
    """Empirical coverage and mean interval width on a held-out test set."""
    y_true = np.asarray(y_true, float).ravel()
    lower = np.asarray(lower, float).ravel()
    upper = np.asarray(upper, float).ravel()
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), 0
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside)), float(np.mean(upper - lower)), n


# ---------------------------------------------------------------------------
# End-to-end wrapper
# ---------------------------------------------------------------------------


def fit_predict_conformal(
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X_proper_train: np.ndarray,
    y_proper_train: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alphas: Sequence[float] = (0.10, 0.05),
) -> Tuple[np.ndarray, Dict[float, ConformalReport]]:
    """Fit, calibrate, and evaluate split-conformal intervals.

    Parameters
    ----------
    fit_predict_fn:
        Callable ``(X_train, y_train, X_eval) -> y_eval_pred`` that fits a
        learner and returns predictions on the eval set.  Wrapping the
        learner this way keeps the conformal logic agnostic to which model
        family is being used.
    X_proper_train, y_proper_train:
        Proper-training set used to fit the base learner.
    X_calibration, y_calibration:
        Calibration set used to compute the residual quantile.  Disjoint
        from the proper-training set by construction.
    X_test, y_test:
        Held-out evaluation set.  ``y_test`` is needed only for coverage
        reporting; it is never seen by the learner or the calibration step.
    alphas:
        Iterable of miscoverage levels.  Each yields one ``ConformalReport``.

    Returns
    -------
    (y_test_pred, reports)
        Point predictions on the test set, and a dict keyed by α with the
        coverage/width/n statistics.
    """
    y_calib_pred = fit_predict_fn(
        np.asarray(X_proper_train, float),
        np.asarray(y_proper_train, float),
        np.asarray(X_calibration, float),
    )
    y_test_pred = fit_predict_fn(
        np.asarray(X_proper_train, float),
        np.asarray(y_proper_train, float),
        np.asarray(X_test, float),
    )

    reports: Dict[float, ConformalReport] = {}
    for alpha in alphas:
        q = split_conformal_quantile(y_calibration, y_calib_pred, alpha)
        lo, hi = conformal_intervals(y_test_pred, q)
        cov, width, n = conformal_coverage(y_test, lo, hi)
        reports[float(alpha)] = ConformalReport(
            alpha=float(alpha),
            coverage=cov,
            mean_width=width,
            n_test=n,
        )
    return y_test_pred, reports


def split_proper_calib(
    n_train: int,
    *,
    calib_fraction: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random index split into proper-training and calibration sets.

    The paper does not specify the proper/calibration ratio explicitly;
    50/50 is the standard default in the conformal literature when no
    domain-specific reason exists to weight one side more heavily.
    """
    if not 0.0 < calib_fraction < 1.0:
        raise ValueError(f"calib_fraction must be in (0, 1); got {calib_fraction}")
    rng = np.random.default_rng(seed)
    idx = np.arange(n_train)
    rng.shuffle(idx)
    n_calib = max(1, int(round(n_train * calib_fraction)))
    return idx[n_calib:], idx[:n_calib]
