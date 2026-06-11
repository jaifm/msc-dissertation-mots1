"""Transfer-learning replication of El-Dalahmeh et al. (Energy and AI 2025).

This subpackage faithfully replicates the LightGBM transfer-learning pipeline
from El-Dalahmeh et al., "Predictive modelling of slurry viscosity using
transfer learning to mitigate uncertainties in pilot-scale lithium-ion battery
manufacturing process" (Energy and AI 22, 100661).

PURPOSE
-------
The dissertation's research question is whether physics-informed ML can
outperform transfer learning for predicting battery slurry rheology.  Answering
that requires a faithful baseline of the published transfer-learning approach
on the same data and under the same evaluation protocol, plus a controlled
audit of the leakage hazards in the original protocol.  This subpackage
provides exactly that.

ISOLATION
---------
Nothing in the main pipeline imports from this subpackage.  The existing
experiment configs (exp_001..exp_026) continue to work unchanged.  To roll
back the replication entirely, remove this folder — no other code will be
affected.

Like the GPR section before it, the runner dispatches to the section entry
point in this subpackage when ``run.sections`` includes ``'tl_baseline'``.
The section consumes the raw train/test splits directly; it does NOT use the
physics-fitted datasets that the rest of the runner produces, because the
paper's pipeline never fits physics models.

PAPER FIDELITY (Phase 1)
------------------------
The implementation matches Section 5 (chained η(10) → η(1)/η(100) strategy),
Section 6 (LightGBM in update mode), Section 7.4 (fine-tuning configuration:
500 base trees at lr=0.10, up to 100 additional trees at lr=0.05), Section 8
(80/20 split + k-fold CV with 95% confidence intervals), and Section 8.1
(split-conformal prediction intervals at 90% and 95%).

The four input features are exactly those listed in Section 2.2:
Formulation (F1, F2, F3 — derived from NMC/C65/KS6L/PVDF percentages per
Table 1), Dispersant_Type (Hypermer, Triton, Tego, none), Solid_Content_pct,
and Solid_Dispersant_pct (= ``Solid_Additive_pct`` in this project's data).

Linear baselines (OLS, Ridge) are included per Section 6.3 to provide the
conservative reference level against which the transfer-learning gain is
quantified.

LEAKAGE AUDIT (Phase 2)
-----------------------
Three controlled deviations from the published protocol are exposed via
config flags so the same code path can run both the faithful replication
and the leakage-controlled variants:

  ``group_aware_split: true``    -- split on Composite_Mix_ID so no
                                    formulation appears in both train and
                                    test (defends against replicate leakage
                                    between Batch_1 and Batch_2)
  ``use_oof_chaining: true``     -- feed out-of-fold η(10) predictions into
                                    the chained η(1)/η(100) models during
                                    training (defends against label-as-feature
                                    leakage in the chained pipeline)
  ``use_chained_strategy: false`` -- predict η(1), η(10), η(100) directly
                                    from composition without chaining (the
                                    "honest baseline" — quantifies how much
                                    of the chained pipeline's lift comes from
                                    the η(10) crutch)

ENTRY POINTS
------------
  ``data``        -- batch-aware loader that derives Formulation and exposes
                     the paper's four-input subset
  ``learners``    -- LightGBM continuation training + OLS/Ridge baselines
                     with a unified fit/predict API
  ``chained``     -- chained η(10) → η(1)/η(100) prediction strategy with
                     in-sample (paper-faithful) and OOF (leakage-controlled)
                     variants
  ``conformal``   -- split-conformal prediction intervals with empirical
                     coverage and width statistics
  ``evaluation``  -- R², MAE, RMSE per shear rate; aggregate across folds
  ``section``     -- ``run_tl_baseline`` entry point used by the runner
"""
from __future__ import annotations

__all__ = [
    "data",
    "learners",
    "chained",
    "conformal",
    "evaluation",
    "section",
]
