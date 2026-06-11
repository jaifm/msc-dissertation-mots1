"""Evaluation metrics matching the paper's reporting protocol.

The paper reports R², MAE, and RMSE per shear rate and per learner, with
fold-level breakdowns aggregated across the k folds of the evaluation
cross-validation.  This module provides:

  * ``score_predictions``  -- compute the three metrics for one prediction
                              vector against its ground truth.
  * ``aggregate_folds``    -- average per-fold metrics for a single
                              (learner, target) combination, with the 95%
                              confidence interval the paper reports.
  * ``build_long_predictions`` -- assemble a long-format DataFrame in the
                              same shape that the rest of the project's
                              SectionResult.predictions consumes, so the
                              notebook plotting code can render TL results
                              without modification.

The metrics are intentionally implemented as standalone functions rather
than reusing ``sklearn.metrics`` so the conformal-coverage column and the
per-fold confidence interval can be computed in one pass without round-
tripping through sklearn's API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Single-prediction-vector scoring
# ---------------------------------------------------------------------------


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        # Constant target — R² is undefined; report nan rather than ±inf.
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float).ravel()
                                - np.asarray(y_pred, float).ravel())))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, float).ravel() - np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean(diff * diff)))


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Return a dict with R², MAE, RMSE for the given prediction vector."""
    return {
        "r2": _r2(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }


# ---------------------------------------------------------------------------
# Per-fold aggregation
# ---------------------------------------------------------------------------


@dataclass
class FoldRow:
    """One row of the per-fold breakdown.

    Used to assemble the ``fold_summary`` DataFrame so each fold's individual
    R²/MAE/RMSE is preserved alongside the aggregate.  Matches the paper's
    "fold means with 95% CI" reporting style.
    """
    fold: int
    learner: str
    variant: str
    target: str
    r2: float
    mae: float
    rmse: float
    n: int


def aggregate_folds(
    fold_rows: Sequence[FoldRow],
    *,
    group_by: Sequence[str] = ("learner", "variant", "target"),
) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean and 95% CI per group.

    The 95% CI is computed as ``mean ± 1.96 * std / sqrt(k)`` from the per-
    fold values within each group — the paper's "fold means with 95% CI"
    construction.  When only one fold is present the half-width is reported
    as 0.0 (rather than nan) to keep downstream tables tidy.
    """
    if not fold_rows:
        return pd.DataFrame(
            columns=list(group_by) + ["r2_mean", "r2_ci95",
                                       "mae_mean", "mae_ci95",
                                       "rmse_mean", "rmse_ci95",
                                       "n_folds"]
        )

    df = pd.DataFrame([r.__dict__ for r in fold_rows])
    out_rows: List[dict] = []
    for keys, sub in df.groupby(list(group_by), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        out: Dict[str, float] = dict(zip(group_by, keys))
        for metric in ("r2", "mae", "rmse"):
            vals = sub[metric].to_numpy(dtype=float)
            out[f"{metric}_mean"] = float(np.nanmean(vals))
            if len(vals) <= 1:
                out[f"{metric}_ci95"] = 0.0
            else:
                std = float(np.nanstd(vals, ddof=1))
                out[f"{metric}_ci95"] = 1.96 * std / np.sqrt(len(vals))
        out["n_folds"] = int(len(sub))
        out_rows.append(out)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Long-format prediction frame
# ---------------------------------------------------------------------------


def build_long_predictions(
    *,
    y_true: Mapping[str, np.ndarray],
    y_pred: Mapping[str, np.ndarray],
    learner: str,
    variant: str,
    fold: Optional[int] = None,
    extra: Optional[Mapping[str, Iterable]] = None,
) -> pd.DataFrame:
    """Build a long-format prediction frame compatible with SectionResult.

    The main pipeline's notebook plots and metric aggregators expect a
    long-format frame with one row per (target, observation) pair plus a
    handful of identifying columns.  This helper reshapes the per-target
    arrays returned by ``predict_chained`` / ``predict_direct`` into that
    schema.

    Parameters
    ----------
    y_true, y_pred:
        Dicts keyed by target column name, each entry a 1-D array.  All
        entries must have the same length.
    learner:
        Learner name (``'lightgbm'`` etc.).  Stored on every row.
    variant:
        Variant identifier — e.g. ``'tl_paper_chained'``,
        ``'tl_oof_chained'``, ``'no_tl_direct'``.  Stored on every row.
    fold:
        Optional integer fold index when iterating over k-fold CV.
    extra:
        Optional dict of extra columns to merge in.  Each iterable must
        be the same length as the target arrays.
    """
    targets = list(y_true.keys())
    if not targets:
        return pd.DataFrame(
            columns=["target", "y_true", "y_pred", "learner", "variant", "fold", "row_idx"]
        )
    n = len(np.asarray(next(iter(y_true.values()))))
    pieces: List[pd.DataFrame] = []
    for tgt in targets:
        yt = np.asarray(y_true[tgt], float).ravel()
        yp = np.asarray(y_pred[tgt], float).ravel()
        if yt.shape[0] != n or yp.shape[0] != n:
            raise ValueError(
                f"length mismatch for target {tgt!r}: "
                f"y_true={yt.shape}, y_pred={yp.shape}, expected n={n}"
            )
        piece = pd.DataFrame({
            "target": tgt,
            "y_true": yt,
            "y_pred": yp,
            "learner": learner,
            "variant": variant,
            "fold": fold if fold is not None else -1,
            "row_idx": np.arange(n, dtype=int),
        })
        if extra:
            for col, vals in extra.items():
                vals = list(vals)
                if len(vals) != n:
                    raise ValueError(
                        f"extra column {col!r} has length {len(vals)}, expected {n}"
                    )
                piece[col] = vals
        pieces.append(piece)
    return pd.concat(pieces, axis=0, ignore_index=True)
