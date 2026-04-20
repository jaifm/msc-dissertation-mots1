"""Per-row physics fits for slurry rheology.

OVERVIEW
--------
Battery electrode slurries are non-Newtonian, shear-thinning fluids: their
viscosity decreases as the applied shear rate increases.  This behaviour
arises because shear breaks down the structural network formed by particles
and binder, and the degree of thinning depends on the formulation.

To predict rheology from composition, we first need a compact numerical
representation of each row's rheology — one that captures the shape of the
viscosity–shear-rate curve rather than storing the raw measurements at three
discrete shear rates.  The physics models here serve that purpose: each model
is fit to the three (shear rate, viscosity) data points and its fitted
parameters become the regression targets.

CHOICE OF MODELS
----------------
Five models are implemented, ranging in flexibility:

  Power Law (2 parameters)
      The simplest empirical model, η = K γ̇^(n−1).  Good for describing
      the power-law region of the flow curve but cannot capture the Newtonian
      plateau at low shear rates.

  Sisko (3 parameters)
      Adds a high-shear Newtonian plateau to the Power Law:
      η = η_∞ + K γ̇^(n−1).  More realistic for slurries at extreme shear.

  Williamson (4 parameters)
      Adds both a zero-shear plateau and a high-shear plateau to give a
      sigmoidal flow curve: η = η_∞ + (η_0 − η_∞) / (1 + (λγ̇)^n).
      Historically used for polymer solutions and particle suspensions.

  Carreau-Yasuda (CY5, 5 parameters)
      The most general model here:
      η = η_∞ + (η_0 − η_∞)(1 + (λγ̇)^a)^((n−1)/a).
      The extra ``a`` parameter controls the rate of transition between the
      two plateaus, making it more flexible than Williamson.

  Constrained CY (3 parameters)
      CY5 with η_∞ fixed at 0.01 Pa·s and ``a`` fixed at 2.0.  Reduces
      the search space for rows where the full 5-parameter optimisation is
      poorly conditioned due to limited data.

With only three viscosity measurements per row, the models with more
parameters are underdetermined in a strict statistical sense.  In practice the
bounds and initial conditions are tight enough to keep the fit meaningful, and
the downstream XGBoost models are evaluated on held-out data, so overfitting
the physics fit is a secondary concern.

FITTING STRATEGY
----------------
All models use ``scipy.optimize.least_squares`` on log-residuals (i.e. we
minimise Σ (log η_fit − log η_measured)²).  The log transform is important
because viscosity spans several orders of magnitude; a linear residual would
let errors at high shear rates dominate and ignore the low-shear behaviour.

Failed fits fall back to the initial parameter vector (p0) rather than
raising an exception, so ``build_physics_dataset`` always gets a complete
row.  The XGBoost models are robust to a small number of poor fits.

NUMERICAL PARITY
----------------
Bounds, initial conditions, and residual formulations are identical to the
original v1 notebook, guaranteeing that physics parameters computed here are
directly comparable to v1 outputs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


def fit_cy5(viscosities: Sequence[float], shear_rates: Sequence[float]) -> np.ndarray:
    """Fit the 5-parameter Carreau-Yasuda model to one row of viscosity data.

    Model: η = η_∞ + (η_0 − η_∞) · (1 + (λγ̇)^a)^((n−1)/a)

    Parameters
    ----------
    viscosities:
        Measured apparent viscosities [Pa·s] at the corresponding shear rates.
    shear_rates:
        Applied shear rates [1/s], same length as ``viscosities``.

    Returns
    -------
    np.ndarray, shape (5,)
        Fitted parameters [eta_0, eta_inf, lambda_time, n, a].
        ``eta_0`` is the zero-shear plateau viscosity [Pa·s].
        ``eta_inf`` is the infinite-shear plateau viscosity [Pa·s].
        ``lambda_time`` is the characteristic relaxation time [s].
        ``n`` is the power-law index (< 1 for shear-thinning).
        ``a`` controls the width of the transition region.
    """
    def residuals(p, visc, sr):
        eta_0, eta_inf, lambda_time, n, a = p
        # Physical constraints: zero-shear must exceed infinite-shear viscosity,
        # all time/shape parameters must be positive.
        if eta_0 <= eta_inf or lambda_time <= 0 or n <= 0 or a <= 0:
            return np.full_like(visc, 1e10)
        eta_fit = eta_inf + (eta_0 - eta_inf) * (1 + (lambda_time * sr) ** a) ** ((n - 1) / a)
        # Log-residuals so errors are scale-invariant across the viscosity range.
        return np.log(eta_fit) - np.log(visc)

    p0 = [10.0, 0.01, 0.1, 0.5, 1.0]
    bounds = ([0.1, 0, 0.01, 0.1, 0.1], [1000, 10, 10, 1.0, 2.0])
    try:
        res = least_squares(
            residuals, p0, args=(np.asarray(viscosities), np.asarray(shear_rates)),
            bounds=bounds, max_nfev=10000,
        )
        return res.x if res.success else np.array(p0)
    except Exception:
        return np.array(p0)


def fit_power_law(viscosities: Sequence[float], shear_rates: Sequence[float]) -> np.ndarray:
    """Fit the 2-parameter Power Law model to one row of viscosity data.

    Model: η = K · γ̇^(n−1)

    The Power Law is the simplest shear-thinning model and is well-suited for
    the mid-range of the flow curve, but it diverges at very low shear rates
    (no zero-shear plateau) and fails at high shear rates (no infinite-shear
    plateau).  It is included as a baseline because its two parameters have
    clear physical interpretations: K is the consistency (higher = thicker
    slurry) and n is the flow behaviour index (lower = more shear-thinning).

    Returns
    -------
    np.ndarray, shape (2,)
        Fitted parameters [k_consistency, n_flow].
    """
    def residuals(p, visc, sr):
        k, n = p
        if k <= 0 or n <= 0:
            return np.full_like(visc, 1e10)
        eta_fit = k * sr ** (n - 1)
        return np.log(eta_fit) - np.log(visc)

    p0 = [1.0, 0.5]
    bounds = ([0.001, 0.01], [1000, 2.0])
    try:
        res = least_squares(
            residuals, p0, args=(np.asarray(viscosities), np.asarray(shear_rates)),
            bounds=bounds, max_nfev=10000,
        )
        return res.x if res.success else np.array(p0)
    except Exception:
        return np.array(p0)


def fit_sisko(viscosities: Sequence[float], shear_rates: Sequence[float]) -> np.ndarray:
    """Fit the 3-parameter Sisko model to one row of viscosity data.

    Model: η = η_∞ + K · γ̇^(n−1)

    Sisko extends the Power Law by adding a high-shear Newtonian plateau
    (η_∞), which is physically meaningful because all real fluids approach a
    finite viscosity at very high shear rates.  It does not include a
    zero-shear plateau, making it most appropriate for slurries measured
    primarily in the shear-thinning region.

    Returns
    -------
    np.ndarray, shape (3,)
        Fitted parameters [eta_inf_sisko, k_sisko, n_sisko].
    """
    def residuals(p, visc, sr):
        eta_inf, k, n = p
        if eta_inf < 0 or k <= 0 or n <= 0:
            return np.full_like(visc, 1e10)
        eta_fit = eta_inf + k * sr ** (n - 1)
        return np.log(eta_fit) - np.log(visc)

    p0 = [0.01, 1.0, 0.5]
    bounds = ([0, 0.001, 0.01], [1, 1000, 2])
    try:
        res = least_squares(
            residuals, p0, args=(np.asarray(viscosities), np.asarray(shear_rates)),
            bounds=bounds, max_nfev=10000,
        )
        return res.x if res.success else np.array(p0)
    except Exception:
        return np.array(p0)


def fit_williamson(viscosities: Sequence[float], shear_rates: Sequence[float]) -> np.ndarray:
    """Fit the 4-parameter Williamson model to one row of viscosity data.

    Model: η = η_∞ + (η_0 − η_∞) / (1 + (λγ̇)^n)

    The Williamson model includes both a zero-shear plateau (η_0) and a
    high-shear plateau (η_∞), giving a sigmoidal flow curve in log–log space.
    The transition is controlled by the relaxation time λ and the exponent n.
    It is structurally similar to CY5 but fixes the transition-width parameter
    at a value equivalent to a = 1, making it less flexible but more stable
    when data are sparse.

    Returns
    -------
    np.ndarray, shape (4,)
        Fitted parameters [eta_0, eta_inf, lambda_time, n_w].
    """
    def residuals(p, visc, sr):
        eta_0, eta_inf, lambda_time, n_w = p
        if eta_0 <= eta_inf or lambda_time <= 0 or n_w <= 0:
            return np.full_like(visc, 1e10)
        eta_fit = eta_inf + (eta_0 - eta_inf) / (1 + (lambda_time * sr) ** n_w)
        return np.log(eta_fit) - np.log(visc)

    p0 = [10.0, 0.01, 0.1, 0.5]
    bounds = ([0.1, 0, 0.01, 0.1], [1000, 10, 10, 1.0])
    try:
        res = least_squares(
            residuals, p0, args=(np.asarray(viscosities), np.asarray(shear_rates)),
            bounds=bounds, max_nfev=10000,
        )
        return res.x if res.success else np.array(p0)
    except Exception:
        return np.array(p0)


def fit_constrained_cy(viscosities: Sequence[float], shear_rates: Sequence[float]) -> np.ndarray:
    """Fit a constrained 3-parameter CY model to one row of viscosity data.

    Model: same as CY5, but with η_∞ = 0.01 Pa·s and a = 2.0 fixed.

    With only three data points per row, the full 5-parameter CY5 optimisation
    is sometimes poorly conditioned: multiple (η_∞, a) combinations can fit
    the data equally well, leading to degenerate solutions.  Fixing η_∞ to a
    physically reasonable high-shear viscosity floor (0.01 Pa·s ≈ 10 mPa·s)
    and the transition width to a = 2.0 (a common literature value) stabilises
    the fit while still capturing the zero-shear plateau (η_0), relaxation
    time (λ), and flow index (n).

    Returns
    -------
    np.ndarray, shape (3,)
        Fitted parameters [eta_0, lambda_time, n].
    """
    def residuals(p, visc, sr):
        eta_0, lambda_time, n = p
        # Fixed parameters: high-shear viscosity floor and transition exponent.
        eta_inf = 0.01
        a = 2.0
        if eta_0 <= eta_inf or lambda_time <= 0 or n <= 0:
            return np.full_like(visc, 1e10)
        eta_fit = eta_inf + (eta_0 - eta_inf) * (1 + (lambda_time * sr) ** a) ** ((n - 1) / a)
        return np.log(eta_fit) - np.log(visc)

    p0 = [10.0, 0.1, 0.5]
    bounds = ([0.1, 0.01, 0.1], [1000, 10, 1.0])
    try:
        res = least_squares(
            residuals, p0, args=(np.asarray(viscosities), np.asarray(shear_rates)),
            bounds=bounds, max_nfev=10000,
        )
        return res.x if res.success else np.array(p0)
    except Exception:
        return np.array(p0)


# Registry mapping config keys to fitter callables.  The keys here must match
# the model identifiers used in ``cfg.physics.models`` (base.yaml) and
# ``cfg.physics.targets``.  Adding a new model requires only: (1) a fitter
# function above, (2) an entry here, and (3) a targets entry in base.yaml.
FITTERS = {
    "cy5": fit_cy5,
    "power_law": fit_power_law,
    "sisko": fit_sisko,
    "williamson": fit_williamson,
    "constrained_cy": fit_constrained_cy,
}
