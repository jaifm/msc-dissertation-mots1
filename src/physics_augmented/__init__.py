"""Physics-augmented training: scipy-fitted curves as a data augmentation oracle.

MOTIVATION
----------
Both the parameter-prediction pipeline (predicting fitted physics parameters
from composition) and the El-Dalahmeh et al. transfer-learning approach
(predicting viscosity at three measured shear rates) under-exploit the data
in different ways:

  * Parameter prediction inherits the under-identifiability of the scipy fit
    (5 parameters from 3 viscosity measurements ⇒ noisy targets).
  * Three-point regression has no notion of shear rate as a continuous
    variable — the model literally cannot answer "what is η at γ̇ = 0.5?".

The physics-augmented protocol sidesteps both:

  1. Train/test split as usual (group-aware on ``Composite_Mix_ID``).
  2. On the **train** rows only, fit a physically-bounded rheological model
     (Power Law or constrained CY) via scipy per row.
  3. Sample N additional shear rates per train row on a log grid in
     [0.1, 1000] s⁻¹ and evaluate the fitted forward model at each.
  4. Stack augmented samples with the three measured points to build a
     long-format ``(composition, γ̇, η)`` training set.
  5. Train a single learner ``f(composition, log γ̇) → η``.
  6. Evaluate on the test set's three measured viscosities — directly
     comparable to TL paper R²/RMSE numbers without any chaining tricks.

The physics model is used as a **teacher/oracle** that interpolates the
curve shape between measurements; the ML student learns composition →
curve mapping over a much denser supervisory signal than 3-point regression
allows.

ISOLATION
---------
Nothing in the main pipeline imports from this subpackage.  The runner
dispatches via the ``physics_augmented`` section key.  To roll back the
methodology, remove this folder — no other code is affected.

ENTRY POINTS
------------
  ``augment``    -- Per-row scipy fit + log-uniform shear sampling + forward
                    evaluation.  Produces the augmented training rows.
  ``data``       -- Builds the long-format ``(composition, γ̇, η)`` train
                    and test DataFrames; exposes a sample-weight column.
  ``learners``   -- XGBoost and GPR wrappers consuming the long-format set
                    and scoring per shear rate on the test split.
  ``section``    -- ``run_physics_augmented`` entry point.
"""
from __future__ import annotations

__all__ = [
    "augment",
    "data",
    "learners",
    "section",
]
