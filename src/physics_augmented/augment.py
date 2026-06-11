"""Per-row scipy fit + log-uniform shear sampling.

Public surface:

  ``fit_row``     -- fit one row's three viscosity measurements with the
                     chosen physics model.  Thin wrapper over the fitters
                     in ``src/physics/models.py``.
  ``forward``     -- evaluate the fitted physics model at one or more
                     shear rates.  Supported models:
                         * ``power_law``
                         * ``constrained_cy``
                         * ``cy5``
                         * ``sisko``
                         * ``williamson``
  ``sample_shears`` -- log-uniform shear grid generator.
  ``augment_train_rows`` -- end-to-end: take a train DataFrame, return a
                     long-format DataFrame with the augmented samples.

LEAKAGE CONTRACT
----------------
The augmentation module deliberately accepts only the *train* DataFrame to
make leakage prevention obvious at the call site.  Test rows are never
fitted or augmented — the test set keeps its raw measured viscosities.
This is the supervisor's standardised methodology (post-May 2026 meeting):

  1. Split the data first (chemistry-based, fixed splits on disk).
  2. Augment only the train side, never the test side.
  3. Evaluate every model on the same fixed test rows.

The dual-helper API (``augment_train_rows`` for train + ``make_test_rows``
for test) keeps these paths visibly distinct in calling code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..physics.models import (
    fit_constrained_cy,
    fit_cy5,
    fit_power_law,
    fit_sisko,
    fit_williamson,
)


# ---------------------------------------------------------------------------
# Forward evaluators (analytic, no scipy)
# ---------------------------------------------------------------------------


def _power_law_forward(params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """η(γ̇) = K · γ̇^(n-1).  ``params`` is [k_consistency, n_flow]."""
    K, n = float(params[0]), float(params[1])
    return K * np.power(np.asarray(gamma, float), n - 1.0)


def _constrained_cy_forward(params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Constrained CY: η_∞ = 0.01, a = 2 fixed.

    η(γ̇) = η_∞ + (η_0 − η_∞) · (1 + (λγ̇)^a)^((n−1)/a)

    ``params`` is [eta_0, lambda_time, n].
    """
    eta_inf = 0.01
    a = 2.0
    eta_0, lam, n = float(params[0]), float(params[1]), float(params[2])
    gamma = np.asarray(gamma, float)
    return eta_inf + (eta_0 - eta_inf) * (1.0 + (lam * gamma) ** a) ** ((n - 1.0) / a)


def _cy5_forward(params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Full 5-parameter CY: η(γ̇) = η_∞ + (η_0−η_∞)·(1+(λγ̇)^a)^((n−1)/a).

    ``params`` is [eta_0, eta_inf, lambda_time, n, a].
    """
    eta_0, eta_inf, lam, n, a = (float(p) for p in params[:5])
    gamma = np.asarray(gamma, float)
    return eta_inf + (eta_0 - eta_inf) * (1.0 + (lam * gamma) ** a) ** ((n - 1.0) / a)


def _sisko_forward(params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Sisko: η(γ̇) = η_∞ + K · γ̇^(n−1).

    ``params`` is [eta_inf_sisko, k_sisko, n_sisko].
    """
    eta_inf, K, n = float(params[0]), float(params[1]), float(params[2])
    gamma = np.asarray(gamma, float)
    return eta_inf + K * np.power(gamma, n - 1.0)


def _williamson_forward(params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Williamson: η(γ̇) = η_∞ + (η_0−η_∞) / (1+(λγ̇)^n).

    ``params`` is [eta_0, eta_inf, lambda_time, n_w].
    """
    eta_0, eta_inf, lam, n = (float(p) for p in params[:4])
    gamma = np.asarray(gamma, float)
    return eta_inf + (eta_0 - eta_inf) / (1.0 + (lam * gamma) ** n)


_FITTERS = {
    "power_law":     fit_power_law,
    "constrained_cy": fit_constrained_cy,
    "cy5":           fit_cy5,
    "sisko":         fit_sisko,
    "williamson":    fit_williamson,
}

_FORWARDS = {
    "power_law":     _power_law_forward,
    "constrained_cy": _constrained_cy_forward,
    "cy5":           _cy5_forward,
    "sisko":         _sisko_forward,
    "williamson":    _williamson_forward,
}


SUPPORTED_MODELS: Tuple[str, ...] = tuple(_FITTERS.keys())


# ---------------------------------------------------------------------------
# Single-row helpers
# ---------------------------------------------------------------------------


def fit_row(
    model_key: str,
    viscosities: Sequence[float],
    shear_rates: Sequence[float],
) -> np.ndarray:
    """Fit one row's three measurements with the chosen physics model."""
    try:
        fitter = _FITTERS[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown physics_augmented model {model_key!r}; "
            f"supported: {SUPPORTED_MODELS}"
        ) from exc
    return fitter(viscosities, shear_rates)


def forward(
    model_key: str,
    params: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Evaluate the chosen physics model at the given shear rates."""
    try:
        fwd = _FORWARDS[model_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown physics_augmented model {model_key!r}; "
            f"supported: {SUPPORTED_MODELS}"
        ) from exc
    return fwd(params, gamma)


def sample_shears(
    n: int,
    shear_range: Tuple[float, float] = (0.1, 1000.0),
    *,
    rng: Optional[np.random.Generator] = None,
    log_uniform: bool = True,
) -> np.ndarray:
    """Generate ``n`` shear rates inside ``shear_range``.

    ``log_uniform`` samples uniformly in log10 space (the default and the
    physically natural choice for shear-rate sweeps).  Setting it False
    falls back to linear-uniform sampling, mostly for ablation.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    lo, hi = float(shear_range[0]), float(shear_range[1])
    if lo <= 0 or hi <= lo:
        raise ValueError(
            f"shear_range must be (lo, hi) with 0 < lo < hi; got {shear_range}"
        )
    if log_uniform:
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        return np.power(10.0, rng.uniform(log_lo, log_hi, size=n))
    return rng.uniform(lo, hi, size=n)


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


@dataclass
class AugmentationStats:
    """Diagnostic counts returned alongside the augmented frame.

    Attributes
    ----------
    n_train_rows:
        Number of input train rows attempted.
    n_fit_success:
        How many rows passed the validity checks below.
    n_skipped_nan_viscosity:
        Rows skipped because one or more measured viscosity was NaN.
    n_skipped_nan_composition:
        Rows skipped because one or more composition feature was NaN.
        Typically only fires on blank/control samples that aren't real
        formulations (e.g. a solvent + dispersant reference measurement
        with no active material).
    n_skipped_invalid_fit:
        Rows where scipy returned a parameter vector that produced
        non-finite or non-positive viscosities on the sample grid.
    """
    n_train_rows: int = 0
    n_fit_success: int = 0
    n_skipped_nan_viscosity: int = 0
    n_skipped_nan_composition: int = 0
    n_skipped_invalid_fit: int = 0


_AUG_COLS_BASE = ["row_id", "shear_rate", "viscosity", "source", "physics_model"]


def augment_train_rows(
    train_df: pd.DataFrame,
    *,
    model_key: str,
    viscosity_columns: Sequence[str],
    measurement_shears: Sequence[float],
    composition_cols: Sequence[str],
    n_aug_points: int = 30,
    shear_range: Tuple[float, float] = (0.1, 1000.0),
    log_uniform: bool = True,
    seed: int = 42,
    row_id_col: str = "Composite_Mix_ID",
) -> Tuple[pd.DataFrame, AugmentationStats]:
    """Produce a long-format augmented training DataFrame.

    For each row in ``train_df``:
      1. Read the three measured viscosities (skipping rows with NaN).
      2. Fit the chosen physics model.
      3. Sample ``n_aug_points`` log-uniform shears in ``shear_range``.
      4. Evaluate the forward model at the sampled shears.
      5. Emit one long-format row per (sampled shear) plus three for the
         original measurements.

    Returns
    -------
    (long_df, stats)
        ``long_df`` has columns:
          * every composition column from ``composition_cols``
          * ``row_id`` (originally ``row_id_col``)
          * ``shear_rate`` (Pa·s⁻¹)
          * ``viscosity`` (Pa·s)
          * ``source`` -- ``"measured"`` or ``"augmented"``
          * ``physics_model`` -- the model_key used for augmentation
        ``stats`` summarises how many rows were processed and skipped.
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown physics_augmented model {model_key!r}; "
            f"supported: {SUPPORTED_MODELS}"
        )
    rng = np.random.default_rng(seed)
    measurement_shears = np.asarray(measurement_shears, float)

    long_rows: List[dict] = []
    stats = AugmentationStats(n_train_rows=int(len(train_df)))
    source_row_counter = 0  # unique integer per accepted training row

    for idx, row in train_df.iterrows():
        try:
            viscosities = np.array(
                [float(row[c]) for c in viscosity_columns], dtype=float
            )
        except (TypeError, ValueError):
            stats.n_skipped_nan_viscosity += 1
            continue
        if not np.all(np.isfinite(viscosities)) or np.any(viscosities <= 0):
            stats.n_skipped_nan_viscosity += 1
            continue

        # Reject rows whose composition features contain NaN.  These are
        # typically blank/control samples (e.g. solvent + dispersant
        # reference measurements with no active material) — the viscosity
        # is real but the composition isn't a valid formulation, so the
        # row can't contribute useful supervision to a composition-to-
        # rheology mapping.
        composition_values = [row.get(c) for c in composition_cols]
        if any(pd.isna(v) for v in composition_values):
            stats.n_skipped_nan_composition += 1
            continue

        params = fit_row(model_key, viscosities, measurement_shears)
        if not np.all(np.isfinite(params)):
            stats.n_skipped_invalid_fit += 1
            continue

        sampled = sample_shears(
            n_aug_points,
            shear_range=shear_range,
            rng=rng,
            log_uniform=log_uniform,
        )
        eta_aug = forward(model_key, params, sampled)
        if not np.all(np.isfinite(eta_aug)) or np.any(eta_aug <= 0):
            # Forward evaluation produced non-physical values somewhere on
            # the sample grid (e.g. Power Law extrapolated to absurd
            # viscosities at γ̇ → 0).  Skip the row's augmentation but keep
            # its measured points so the row is still represented.
            stats.n_skipped_invalid_fit += 1
            sampled = np.empty(0)
            eta_aug = np.empty(0)
        else:
            stats.n_fit_success += 1

        row_identifier = row.get(row_id_col, idx)
        composition = {c: row.get(c) for c in composition_cols}
        src_idx = source_row_counter
        source_row_counter += 1

        # Measured points.
        for gam, eta in zip(measurement_shears.tolist(), viscosities.tolist()):
            long_rows.append({
                **composition,
                "row_id": row_identifier,
                "source_row_idx": src_idx,
                "shear_rate": float(gam),
                "viscosity": float(eta),
                "source": "measured",
                "physics_model": model_key,
            })

        # Augmented points.
        for gam, eta in zip(sampled.tolist(), eta_aug.tolist()):
            long_rows.append({
                **composition,
                "row_id": row_identifier,
                "source_row_idx": src_idx,
                "shear_rate": float(gam),
                "viscosity": float(eta),
                "source": "augmented",
                "physics_model": model_key,
            })

    long_df = pd.DataFrame(long_rows)
    return long_df, stats


def make_test_rows(
    test_df: pd.DataFrame,
    *,
    viscosity_columns: Sequence[str],
    measurement_shears: Sequence[float],
    composition_cols: Sequence[str],
    row_id_col: str = "Composite_Mix_ID",
) -> pd.DataFrame:
    """Build the long-format test DataFrame (measured rows only).

    Test rows are NEVER fitted or augmented — that would be leakage.
    This helper just reshapes the wide test DataFrame so that scoring can
    use the same ``(composition, shear_rate, viscosity)`` schema as
    training.
    """
    measurement_shears = list(measurement_shears)
    long_rows: List[dict] = []
    for idx, row in test_df.iterrows():
        try:
            viscosities = [float(row[c]) for c in viscosity_columns]
        except (TypeError, ValueError):
            continue
        composition = {c: row.get(c) for c in composition_cols}
        # Same blank-row guard as ``augment_train_rows``: if any composition
        # feature is NaN, the model can't learn from or be scored on it.
        if any(pd.isna(v) for v in composition.values()):
            continue
        row_identifier = row.get(row_id_col, idx)
        for gam, eta in zip(measurement_shears, viscosities):
            if not np.isfinite(eta) or eta <= 0:
                continue
            long_rows.append({
                **composition,
                "row_id": row_identifier,
                "shear_rate": float(gam),
                "viscosity": float(eta),
                "source": "measured",
                "physics_model": "n/a",
            })
    return pd.DataFrame(long_rows)
