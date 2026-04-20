"""Physically motivated composition features for rheology prediction.

OVERVIEW
--------
The raw composition columns describe what is in a battery slurry (NMC active
material, conductive additives C65 and KS6L, PVDF binder, solid content, and
a small additive like a dispersant) but they do not encode how those components
interact to produce flow behaviour.  Slurry rheology is governed by a small
number of structural mechanisms:

  1. **Particle crowding** — as solids loading increases, particles are forced
     into closer contact.  Viscosity rises sharply near the jamming limit.

  2. **Percolating conductive network** — carbon black particles (C65, KS6L)
     form a stress-bearing network at relatively low concentrations.  The
     combined conductive loading, not the individual fractions alone, determines
     whether a network exists.

  3. **Binder bridging** — PVDF dissolves in NMP solvent and adsorbs onto
     particle surfaces.  It acts as both a lubricant (at low concentrations) and
     a bridging agent that raises zero-shear viscosity (at high concentrations).
     Ratios of particle mass to binder mass capture this competition.

  4. **Shear-thinning magnitude** — the degree of structure breakdown under
     shear scales with how much structure exists at rest, which is jointly
     controlled by conductive network strength and overall solids crowding.

The functions below encode these mechanisms as explicit scalar features so a
tree-based model (XGBoost) can learn the relevant interactions without needing
to discover them from hundreds of product terms.

All columns are guarded: a feature is only added when every required input
column is present in the supplied DataFrame.  This makes the function safe to
call on any column subset without error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Small epsilon added to denominators to prevent division-by-zero when a
# component is absent from a formulation (e.g. PVDF_pct == 0).
_EPS = 1e-6


def engineer_composition_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with engineered features appended.

    The input columns are preserved unchanged; new feature columns are added
    after the last original column.  Column names follow the pattern
    <descriptor>_<units_or_suffix> so they are self-documenting.

    Parameters
    ----------
    X_df:
        Composition DataFrame whose columns are a subset of the raw composition
        columns defined in the project config (composition_cols).

    Returns
    -------
    pd.DataFrame
        Copy of ``X_df`` with additional feature columns appended.
    """
    engineered = X_df.copy()

    # ------------------------------------------------------------------
    # GROUP 1 — Additive totals
    #
    # Individual component percentages lose information about how additives
    # combine.  C65 (carbon black) and KS6L (graphite flake) both contribute
    # to the electronic conductive network inside the electrode, and their
    # combined loading is the relevant quantity for network percolation theory.
    # Similarly, total non-binder solids is a measure of how crowded the
    # dispersion is before accounting for the binder's swelling contribution.
    # ------------------------------------------------------------------

    if {"C65_pct", "KS6L_pct"}.issubset(engineered.columns):
        # Sum of both carbon additive fractions.  At a critical combined
        # loading (~2 wt% for carbon black) the particles form a percolating
        # network that dramatically increases zero-shear viscosity and elastic
        # modulus.  Using the total rather than each fraction separately lets
        # the model capture this threshold regardless of the C65/KS6L split.
        engineered["conductive_total"] = engineered["C65_pct"] + engineered["KS6L_pct"]
    else:
        # Placeholder NaN so downstream engineered features that depend on
        # conductive_total still have a column to reference.
        engineered["conductive_total"] = np.nan

    if {"NMC_pct", "C65_pct", "KS6L_pct"}.issubset(engineered.columns):
        # Total solid loading that is NOT binder.  This is closely related to
        # the hard-sphere effective volume fraction: as it approaches ~80 wt%
        # (depending on particle size distribution), the slurry transitions
        # from a viscous suspension to a paste-like material with yield stress.
        engineered["nonbinder_total"] = (
            engineered["NMC_pct"] + engineered["C65_pct"] + engineered["KS6L_pct"]
        )
    else:
        engineered["nonbinder_total"] = np.nan

    if "PVDF_pct" in engineered.columns:
        # Alias included explicitly so models trained on engineered features
        # always see a column named binder_share, even if PVDF_pct is also
        # present.  This avoids ambiguity in the feature importance rankings.
        engineered["binder_share"] = engineered["PVDF_pct"]

    # ------------------------------------------------------------------
    # GROUP 2 — Particle-to-binder ratios
    #
    # Ratios encode the competition between particle crowding and binder
    # coverage.  A high NMC-to-binder ratio means the binder layer on each
    # particle is thin; inter-particle contact is more likely and the network
    # is stronger.  A high conductive-to-binder ratio captures whether the
    # binder can fully coat the high-surface-area carbon additives (which have
    # much larger specific surface area than NMC).
    #
    # These ratios appear in classical rheology literature for concentrated
    # suspensions (e.g. Krieger-Dougherty scaling) as well as in battery
    # slurry formulation guides.
    # ------------------------------------------------------------------

    if {"NMC_pct", "PVDF_pct"}.issubset(engineered.columns):
        # Ratio of active material mass to binder mass.  Increasing this ratio
        # (less binder per unit of NMC) tends to reduce the viscosity plateau
        # but increase brittleness in the dried electrode — here it is a
        # predictor of the wet-slurry rheology, not the electrode quality.
        engineered["NMC_to_binder"] = engineered["NMC_pct"] / (engineered["PVDF_pct"] + _EPS)

    if {"conductive_total", "PVDF_pct"}.issubset(engineered.columns):
        # Ratio of combined conductive additive mass to binder mass.
        # Carbon black has a BET surface area ~60× larger than NMC, so it
        # preferentially adsorbs PVDF.  When this ratio is high, the binder
        # is consumed by the carbon network, leaving less to coat NMC and
        # acting as an effective cross-linker between carbon particles.
        engineered["conductive_to_binder"] = (
            engineered["conductive_total"] / (engineered["PVDF_pct"] + _EPS)
        )

    if {"Solid_Content_pct", "Solid_Additive_pct"}.issubset(engineered.columns):
        # Ratio of total dry solid content to the small liquid/surface additive
        # (e.g. Hypermer dispersant).  A high ratio means fewer additive
        # molecules per unit of particle surface area, which affects steric
        # stabilisation and therefore the strength of the electrostatic/steric
        # repulsion that keeps the slurry from flocculating.
        engineered["solid_to_additive"] = (
            engineered["Solid_Content_pct"] / (engineered["Solid_Additive_pct"] + _EPS)
        )

    # ------------------------------------------------------------------
    # GROUP 3 — Interaction (product) terms
    #
    # Tree-based models split on one feature at a time and can approximate
    # interactions, but only with many splits across both features.  Explicit
    # product terms make the interaction a first-class feature so it can be
    # captured with a single split.
    #
    # Physical motivation: all three interactions below share the pattern
    # "overall crowding × local structure driver".  Viscosity depends not just
    # on how much solid is present (Solid_Content_pct) but on how that solid
    # is organised — dominated by large NMC spheres (solid_x_nmc), lubricated
    # by binder (solid_x_pvdf), or stiffened by the carbon network
    # (solid_x_conductive).  The product captures the joint effect.
    # ------------------------------------------------------------------

    if {"Solid_Content_pct", "NMC_pct"}.issubset(engineered.columns):
        # Captures crowding of the primary active-material particles.  At fixed
        # NMC_pct, higher solid content means more total particles; at fixed
        # solid content, higher NMC_pct means larger dominant particles.  Both
        # together determine how severely the suspension approaches jamming.
        engineered["solid_x_nmc"] = engineered["Solid_Content_pct"] * engineered["NMC_pct"]

    if {"Solid_Content_pct", "PVDF_pct"}.issubset(engineered.columns):
        # Captures the binder concentration in the context of overall loading.
        # Binder has opposite effects at different solids loadings: at low
        # solids it is primarily a lubricant (reduces viscosity), but at high
        # solids it bridges particles and increases zero-shear viscosity.  The
        # product term helps the model identify which regime applies.
        engineered["solid_x_pvdf"] = engineered["Solid_Content_pct"] * engineered["PVDF_pct"]

    if {"Solid_Content_pct", "conductive_total"}.issubset(engineered.columns):
        # Captures how strongly the conductive network reinforces the
        # suspension structure.  The network's contribution to viscosity is
        # amplified when the background suspension is already crowded (high
        # solid content), because denser packing constrains particle rearrangement
        # and makes the network harder to break under shear.
        engineered["solid_x_conductive"] = (
            engineered["Solid_Content_pct"] * engineered["conductive_total"]
        )

    # ------------------------------------------------------------------
    # GROUP 4 — Log-stabilised small-fraction columns
    #
    # Solid_Additive_pct, C65_pct, and KS6L_pct are typically small (0–5 wt%)
    # and right-skewed: most formulations use a small dose, a few use a larger
    # dose.  Raw percentages give very little spread at the low end, making it
    # hard for tree splits to distinguish between 0.1 % and 0.3 % additive —
    # but that small difference can change the particle dispersion state
    # significantly.  log1p (== log(1 + x)) compresses the high end and
    # stretches the low end, making fine distinctions at small doses visible
    # to the model without requiring extra binning or custom splits.
    # ------------------------------------------------------------------

    for col in ("Solid_Additive_pct", "C65_pct", "KS6L_pct"):
        if col in engineered.columns:
            # log1p is used instead of log to handle exact-zero values (e.g.
            # formulations with no KS6L) without producing -inf.
            engineered[f"log1p_{col}"] = np.log1p(engineered[col])

    return engineered
