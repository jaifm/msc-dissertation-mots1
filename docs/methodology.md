# Methodology — How the Project Evolved

This document narrates the project's methodological evolution in the order
it actually happened, including the dead ends.  Each stage states the
question, the method, the result, and what forced the next stage.  The
repository's commit history mirrors this narrative.

## The deployment problem

Given the composition of a new battery-cathode slurry, predict its full
viscosity–shear-rate curve before any wet-lab measurement.  The model must
be queryable at *any* shear rate in the industrial 0.1–1000 s⁻¹ window —
not only at the three rates (1, 10, 100 s⁻¹) where the training data was
measured.  This requirement drives every design decision below.

**Data.** 177 rows after cleaning; 67 unique chemistries
(`Composite_Mix_ID` = Formulation × Dispersant × Solid_Content ×
Solid_Additive); two collection campaigns (`Source_Batch`); three measured
viscosities per row.  One solvent-only blank row is filtered everywhere.

## Stage 1 — Composition → physics parameters (Approach 1)

**Question.** Can we predict the parameters of a fitted rheological model
(Power Law, Sisko, Williamson, Carreau–Yasuda) from composition, then
reconstruct the curve through the forward equation?

**Method.** Per-row scipy `least_squares` fits in log-residual space
produce parameter targets; XGBoost (Optuna-tuned) or GPR regress them
from composition.  Sections: `baseline`, `sweep`, `loo`, `gpr_baseline`.

**Result.** Plateaued.  The consistency parameter K is moderately
learnable (R² ≈ 0.65); the flow index n is not (R² ≈ 0.31), and the
within/cross-model gap analysis showed the ceiling is set by the
under-determined three-point fit producing noisy parameter targets — not
by the ML student.  *n cannot be reliably inferred from composition for
genuinely unseen recipes*; this is the project's core bottleneck finding.

## Stage 2 — Transfer-learning replication and audit (Approach 2)

**Question.** Does the published TL approach (El-Dalahmeh et al. 2025,
LightGBM continuation training Batch_1 → Batch_2) solve the problem?

**Method.** Faithful replication (`src/transfer_learning/`), then a
leakage audit: in-sample vs out-of-fold chaining, row-level vs
group-aware splits, oracle-chain variants, conformal intervals.

**Result.** The paper's R² ≈ 0.85–0.92 is not reproducible under any
honest protocol (best faithful variant 0.81 with in-sample chaining
leakage; honest variants 0.59–0.77).  Three audit findings: the no-TL
baseline never sees Batch_2 (inflating the TL gain), chained prediction
leaks in-sample, and η(1)'s heavy tail makes the claimed raw-space R²
implausible without an unstated transform.  Structurally, the TL
pipeline is shear-blind: γ̇ is not an input, so it cannot answer the
deployment question at all.

## Stage 3 — Physics-augmented training (Approach 3)

**Question.** Can the physics fit be used as an *interpolator* rather
than a target, sidestepping the noisy-parameter bottleneck?

**Method.** Fit each *training* row's three points; sample the fitted
curve at 30 log-uniform shears; train one student
`f(composition, log γ̇) → log1p η` on the expanded set.  Augmentation is
train-only (leakage contract).  Log-space R² is the primary metric (raw
R² is dominated by the η(1) scale).

**Results (XGBoost student).**
- Random fixed split: R²(log) = 0.815 — but identical to the
  no-augmentation control (exp_033), falsifying the naive "more data"
  hypothesis at the measured shears.
- The held-out-shear study reframed the contribution: train on two
  shears, test on the third.  Without augmentation the tree student
  collapses (R²(log) −2.6 at η(10)); with it, +0.68.  Augmentation
  teaches curve *shape*, which a tree cannot interpolate on its own.

## Stage 4 — Standardised evaluation (chemistry splits)

After a supervisor decision, all headline evaluation moved to fixed,
hash-pinned splits keyed on the full recipe (`Composite_Mix_ID`):
default ~25% stratified hold-out, 67 leave-one-chemistry-out folds,
few-shot (k ∈ {1,2}) variants, and two cross-batch splits.  The honest
strict-hold-out headline became R²(log) = 0.753 (vs 0.815 random-split —
the 0.06 gap *quantifies recipe-level leakage* in random splits).
Few-shot showed one measurement of a new recipe recovers most of the
gap; two approach the fixed-split ceiling.

## Stage 5 — Physics priors and the linear student (improvement round)

Four literature-backed extensions were tested on identical splits:

1. **Monotone shear-thinning constraint** (exp_040): no gain at measured
   shears (0.735); improves curve shape above γ̇=10, degrades it below
   γ̇=1.  Reported as a trade-off.
2. **Maron–Pierce crowding feature** (exp_041/042): no gain for the tree
   student (0.740 / 0.716), consistent with exp_038's engineered-feature
   null result.
3. **GPR student** (exp_043): matches XGBoost (0.752), adds curve-level
   predictive variance.
4. **Varying-coefficient ridge student** (exp_044): **R²(log) = 0.774**
   — the new headline.  Motivated by the TL head-to-head study, where a
   per-shear ridge in log space outperformed every tree model: trees
   predict constants outside the training composition hull, while
   log-viscosity is near-linear in composition (Mooney-type laws).  The
   ridge student is linear in `[x, log γ̇, x·log γ̇]` — a power-law
   surface with composition-dependent consistency and flow index — so it
   extrapolates in composition *and* remains queryable at any shear.
   It also: predicts held-out shears without augmentation (the right
   inductive bias built in), leads most curve-shape bands, and is
   the best-calibrated model under conformal evaluation.

**Revised augmentation story.** Augmentation is structural for students
without a shear-direction inductive bias (trees), and a modest
high-shear-extrapolation aid for students with one (ridge).  This
decomposition — bias repair vs information addition — is the
dissertation's sharpest framing of what physics augmentation does.

## Stage 6 — Head-to-head under identical evaluation (RQ2 answer)

The `tl_chemistry` study transplants the paper's TL mechanism onto the
chemistry split (pretrain on the train rows from Batch_1, fine-tune on
the train rows from Batch_2, identical held-out recipes):

| Model (log1p target) | Pooled R²(log) | Any-shear? |
|---|---|---|
| Physics-augmented ridge (exp_044) | **0.774** | yes |
| Physics-augmented XGBoost (exp_035) | 0.753 | yes |
| Physics-augmented GPR (exp_043) | 0.752 | yes |
| Per-shear ridge (4 inputs) | 0.811 | **no** |
| No-TL pooled LightGBM | 0.625 | no |
| TL LightGBM (paper mechanism) | 0.574 | no |

Physics-informed ML beats transfer learning on identical test rows, and
the only model that edges out the ridge student (the per-shear ridge)
cannot answer the deployment question.

## Retired direction — viscosity-space loss

A custom XGBoost objective scoring predicted parameters against measured
viscosities (JAX-differentiated) was prototyped and removed: the loss
only sees the three measured shears, providing no incentive for the
curve to be sensible elsewhere.  Documented to prevent re-proposal.

## Research questions → where they are answered

1. *Composition-only prediction at any shear rate?*  Held-out-shear study
   (`shear_holdout`, both learners) + oracle curve evaluation.
2. *Physics-informed ML vs transfer learning?*  `tl_chemistry` study on
   identical chemistry-split rows (table above).
3. *Where does the error come from?*  Within/cross gap (Approach 1),
   LOO per-formulation breakdown, the augmentation decomposition (§5).
4. *How much do random splits inflate claims?*  0.815 → 0.753
   (random vs chemistry split), 0.06 of pure recipe leakage.
