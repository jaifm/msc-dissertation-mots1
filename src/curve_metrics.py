"""Curve-space RMSE: evaluate ML predictions in viscosity-curve space.

WHY CURVE-SPACE RMSE?
---------------------
Parameter-space R² is biased toward simple models.  Power Law has only two
parameters, so ML can identify them precisely — but the Power Law curve has
no zero-shear plateau and diverges as γ̇ → 0, making it wrong for deployment
even when R² is high.  Curve-space RMSE exposes this by comparing the actual
viscosity curve produced by the ML-predicted parameters against a reference
curve on a dense shear grid.

TWO VARIANTS
------------
cross_model
    Reference = Constrained CY (η_∞=0.01, a=2 fixed) fitted to the same three
    raw viscosity measurements via scipy.  Measures absolute deployment value:
    how close is the ML-predicted curve to the best identifiable reference for
    this slurry?  Penalises models whose functional form diverges from the
    reference regardless of how well the ML recovers the parameters.

within_model
    Reference = the scipy-fitted curve for the same model (i.e. ground-truth
    parameter values already stored in the physics dataset).  Measures ML
    quality only: did the ML recover the model's own parameters accurately?
    Low within-model RMSE with high cross-model RMSE → ML is good but the model
    family is wrong.

METRIC
------
Log-RMSE: sqrt(mean((log η_pred − log η_ref)²)) over the dense shear grid.
Log scale is used because viscosity spans several orders of magnitude; an
arithmetic RMSE would be dominated by high-viscosity (low shear) differences.
Units: dimensionless (log Pa·s).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .config import Config
from .physics.models import fit_constrained_cy

_CONSTRAINED_CY_ETA_INF = 0.01
_CONSTRAINED_CY_A = 2.0
_DEFAULT_SHEAR_GRID = np.logspace(np.log10(0.1), np.log10(1000.0), 200)


# ---------------------------------------------------------------------------
# Curve evaluation
# ---------------------------------------------------------------------------


def _eval_curve(model_key: str, params: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Evaluate a rheology model's viscosity curve at shear rates ``gamma``.

    Parameter order must match ``cfg.physics.targets[model_key]`` exactly.
    """
    if model_key == "power_law":
        k, n = params
        return k * gamma ** (n - 1)

    if model_key == "sisko":
        eta_inf, k, n = params
        return eta_inf + k * gamma ** (n - 1)

    if model_key == "williamson":
        eta_0, eta_inf, lam, n = params
        return eta_inf + (eta_0 - eta_inf) / (1 + (lam * gamma) ** n)

    if model_key == "cy5":
        eta_0, eta_inf, lam, n, a = params
        return eta_inf + (eta_0 - eta_inf) * (1 + (lam * gamma) ** a) ** ((n - 1) / a)

    if model_key == "constrained_cy":
        eta_0, lam, n = params
        return (
            _CONSTRAINED_CY_ETA_INF
            + (eta_0 - _CONSTRAINED_CY_ETA_INF)
            * (1 + (lam * gamma) ** _CONSTRAINED_CY_A) ** ((n - 1) / _CONSTRAINED_CY_A)
        )

    raise ValueError(f"Unknown model_key for curve evaluation: {model_key!r}")


def _log_rmse(eta_pred: np.ndarray, eta_ref: np.ndarray) -> float:
    """RMSE in log-viscosity space.  Returns NaN if any value is non-positive."""
    if np.any(eta_pred <= 0) or np.any(eta_ref <= 0):
        return np.nan
    return float(np.sqrt(np.mean((np.log(eta_pred) - np.log(eta_ref)) ** 2)))


# ---------------------------------------------------------------------------
# Per-model curve RMSE
# ---------------------------------------------------------------------------


def compute_model_curve_rmse(
    model_key: str,
    model_preds_long: pd.DataFrame,
    test_physics: pd.DataFrame,
    cfg: Config,
    shear_grid: Optional[np.ndarray] = None,
) -> dict:
    """Compute cross-model and within-model curve-space RMSE for one ML model.

    Parameters
    ----------
    model_key:
        Physics model identifier (e.g. ``'power_law'``, ``'cy5'``).
    model_preds_long:
        Long-format predictions DataFrame for this model.  Must contain columns
        ``row_idx``, ``y_pred``, and ``target``.  Typically built by
        ``pd.concat`` of the per-target ``predictions_df`` returned by
        ``fit_and_evaluate_gpr`` or ``tune_and_evaluate_xgb``.
    test_physics:
        Test-split physics dataset.  Used for (a) raw viscosity columns needed
        to fit the Constrained CY reference and (b) scipy-fitted target columns
        used as the within-model reference.
    cfg:
        Experiment configuration.  Provides ``cfg.physics.targets[model_key]``
        (parameter column order) and ``cfg.data.viscosity_columns`` / shear
        rates for fitting the reference.
    shear_grid:
        Dense shear rate array [1/s] on which curves are evaluated.  Defaults
        to 200 log-spaced points from 0.1 to 1000 1/s.

    Returns
    -------
    dict with keys:
        ``mean_curve_rmse_cross``  -- mean log-RMSE vs Constrained CY reference
        ``mean_curve_rmse_within`` -- mean log-RMSE vs within-model scipy fit
        ``per_row``                -- DataFrame[row_idx, curve_rmse_cross,
                                       curve_rmse_within] for per-row diagnostics
    """
    if shear_grid is None:
        shear_grid = _DEFAULT_SHEAR_GRID

    target_cols = cfg.targets_for(model_key)
    visc_cols = cfg.data.viscosity_columns
    shear_rates = np.array(cfg.physics.viscosity_shears)

    # Pivot long predictions to wide: index=row_idx, columns=target names
    preds_wide = model_preds_long.pivot(index="row_idx", columns="target", values="y_pred")

    # Align scipy-fitted params (y_true) from test_physics by positional index
    true_wide = test_physics[target_cols].reset_index(drop=True)
    raw_visc = test_physics[visc_cols].reset_index(drop=True)

    n_rows = len(preds_wide)
    cross_rmses: list = []
    within_rmses: list = []
    row_indices: list = list(preds_wide.index)
    pred_viscs: list = []  # predicted viscosity at the 3 measured shear rates
    true_viscs: list = []  # actual measured viscosity at those shear rates

    for pos in range(n_rows):
        row_key = preds_wide.index[pos]

        try:
            pred_params = preds_wide.loc[row_key, target_cols].to_numpy(dtype=float)
        except Exception:
            cross_rmses.append(np.nan)
            within_rmses.append(np.nan)
            continue

        true_params = true_wide.iloc[pos].to_numpy(dtype=float)
        visc_vals = raw_visc.iloc[pos].to_numpy(dtype=float)

        # ML-predicted viscosity curve
        try:
            pred_curve = _eval_curve(model_key, pred_params, shear_grid)
        except Exception:
            cross_rmses.append(np.nan)
            within_rmses.append(np.nan)
            continue

        # Cross-model reference: Constrained CY fitted to the 3 raw measurements
        try:
            ref_params_cross = fit_constrained_cy(visc_vals, shear_rates)
            ref_curve_cross = _eval_curve("constrained_cy", ref_params_cross, shear_grid)
            cross_rmses.append(_log_rmse(pred_curve, ref_curve_cross))
        except Exception:
            cross_rmses.append(np.nan)

        # Within-model reference: scipy-fitted params already in test_physics
        try:
            ref_curve_within = _eval_curve(model_key, true_params, shear_grid)
            within_rmses.append(_log_rmse(pred_curve, ref_curve_within))
        except Exception:
            within_rmses.append(np.nan)

        # Viscosity R²: evaluate the predicted curve at the 3 measured shear
        # rates and compare directly against the raw measurements.  This gives
        # a metric on the same scale as the TL paper's per-shear-rate R².
        try:
            pred_at_measured = _eval_curve(model_key, pred_params, shear_rates)
            if np.all(pred_at_measured > 0) and np.all(visc_vals > 0):
                pred_viscs.append(pred_at_measured)
                true_viscs.append(visc_vals)
        except Exception:
            pass

    per_row = pd.DataFrame({
        "row_idx": row_indices,
        "curve_rmse_cross": cross_rmses,
        "curve_rmse_within": within_rmses,
    })

    # Viscosity-space R² at each of the 3 measured shear rates.
    visc_r2 = float("nan")
    visc_r2_per_shear: list = [float("nan")] * len(shear_rates)
    if pred_viscs:
        pred_arr = np.array(pred_viscs)  # shape [n, 3]
        true_arr = np.array(true_viscs)
        visc_r2 = float(r2_score(true_arr.ravel(), pred_arr.ravel()))
        visc_r2_per_shear = [
            float(r2_score(true_arr[:, i], pred_arr[:, i]))
            for i in range(true_arr.shape[1])
        ]

    return {
        "mean_curve_rmse_cross": float(np.nanmean(cross_rmses)),
        "mean_curve_rmse_within": float(np.nanmean(within_rmses)),
        "visc_r2": visc_r2,
        "visc_r2_shear_rates": shear_rates.tolist(),
        "visc_r2_per_shear": visc_r2_per_shear,
        "per_row": per_row,
    }
