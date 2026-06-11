# Experiment Results Reference

Key findings per experiment.  All R² values are on the fixed chemistry split
(`splits_chemistry/default`, 16 held-out Composite_Mix_IDs, n_test=132)
unless otherwise noted.  **R²(log)** is the primary metric — computed in
log1p space, fair across all shear rates.  R²(raw) is retained for
comparison only.

Regeneration commands for every table: [docs/reproducibility.md](docs/reproducibility.md).
Literature backing for every methodological idea: [docs/literature.md](docs/literature.md).

---

## Metric note

Raw-space R² is dominated by η(1) Pa·s values (~100× larger than η(100)),
making it a biased metric that rewards fitting low-shear viscosity at the
expense of high-shear accuracy.  Log-space R² is the correct primary metric.
All headline claims in the dissertation use R²(log).

---

## HEADLINE — model comparison on identical chemistry-split test rows

| Model | Experiment / study | Pooled R²(log) | Any-shear capable? |
|---|---|---|---|
| **Physics-augmented ridge (varying-coefficient)** | exp_044 | **0.802** | **yes** |
| Per-shear ridge, 4 paper inputs | tl_chemistry | 0.811 | no |
| Physics-augmented XGBoost | exp_035 | 0.753 | yes |
| Physics-augmented GPR | exp_043 | 0.752 | yes |
| Physics-augmented XGBoost + monotone prior | exp_040 | 0.735 | yes |
| No-TL pooled LightGBM, 4 paper inputs | tl_chemistry | 0.625 | no |
| TL LightGBM (El-Dalahmeh mechanism) | tl_chemistry | 0.574 | no |
| Parameter-prediction GPR (Approach 1) | exp_037 | 0.166* | yes (via params) |
| Parameter-prediction XGBoost (Approach 1) | exp_037 | 0.014* | yes (via params) |

\* mean parameter R², not pooled viscosity R²(log) — see exp_037 section.

The only model that edges out the physics-augmented ridge is the per-shear
ridge, which trains one model per measured shear and cannot be queried at
any other shear rate — it cannot answer the deployment question.

---

## Transfer Learning (El-Dalahmeh et al. replication)

### exp_027 — TL replication

| Variant | R²(mean) | η(1) R² | η(10) R² | η(100) R² |
|---|---|---|---|---|
| tl_paper_chained (paper-faithful) | 0.722 | 0.486 | 0.811 | 0.870 |
| tl_paper_chained_oof (OOF chain) | 0.773 | 0.640 | 0.811 | 0.870 |
| tl_paper_direct (no chain) | 0.772 | 0.649 | 0.811 | 0.856 |
| no_tl_chained (Batch_2 only) | 0.737 | 0.415 | **0.878** | **0.916** |
| tl_paper_chained_groups (group-aware) | **0.599** | 0.390 | 0.637 | 0.769 |

Key findings:
- In-sample chaining inflates η(1) R² by +0.15 (0.486 → 0.640 with OOF).
- Group-aware split drops overall R² from 0.722 → 0.599 — the 0.12 gap is
  pure Composite_Mix_ID leakage (same recipe in train and test).
- Training on Batch_2 alone (no TL) achieves 0.737 — comparable to TL.
- The paper's claimed R²≈0.85–0.92 was not reproduced under any honest protocol.

### exp_028 — TL reproduction attempt

| Variant | R²(mean) |
|---|---|
| paper_tl_chained | 0.638 |
| paper_no_tl_baseline | 0.638 (identical to TL — fine-tuning had no effect) |
| batch2_only_chained | 0.567 |
| paper_tl_chained_oof | 0.589 |

Key finding: TL fine-tuning provided zero measurable benefit over the no-TL
baseline.  The paper's numbers are not reproducible.

### exp_036 — Oracle-chain faithful reproduction (true-labels chaining)

Most aggressive faithful interpretation: measured η(10) used as chaining
feature during training (rather than model predictions).  K-fold on full
Batch_2, raw Pa·s, paper hyperparameters (500/100 trees, lr=0.10/0.05).

| Variant | η(1) R² | η(10) R² | η(100) R² | Mean R² |
|---|---|---|---|---|
| true_labels_tl (oracle chain + TL) | 0.520 | 0.810 | 0.834 | 0.721 |
| insample_tl (paper-faithful, in-sample) | **0.751** | 0.810 | **0.881** | **0.814** |
| oof_tl (honest OOF chain + TL) | 0.591 | 0.810 | 0.862 | 0.754 |
| true_labels_no_tl (oracle chain, no TL) | −2.94 | 0.121 | 0.314 | −0.836 |

KEY FINDING: The oracle chain (use_true_labels=True) is **worse** than
in-sample chaining on η(1) R² (0.520 vs 0.751).  Reason: training with
perfect measured η(10) creates a distribution mismatch at test time where
the model receives noisy predicted η(10) (R²≈0.81).  The paper's η(1)
R² ≈ 0.85 is unreproducible under every chaining interpretation attempted;
the 0.10+ gap to the claim cannot be closed by any chaining strategy.

### tl_chemistry study — TL vs physics augmentation on IDENTICAL test rows

The paper's TL mechanism transplanted onto the chemistry split: pretrain
on the chemistry-train rows from Batch_1 (500 trees, lr 0.10), fine-tune
on the chemistry-train rows from Batch_2 (100 trees, lr 0.05), evaluate
on the same 16 held-out recipes as exp_035/exp_044.  Paper's 4-input
feature set; one model per measured shear (the TL paradigm is
shear-blind).  CSV: `tl_chemistry_results.csv`.

| Variant (log1p target) | η(1) R²(log) | η(10) R²(log) | η(100) R²(log) | Pooled R²(log) |
|---|---|---|---|---|
| tl_lightgbm | 0.259 | 0.438 | 0.679 | 0.574 |
| pooled_lightgbm (no TL) | 0.344 | 0.531 | 0.677 | 0.625 |
| ridge (per-shear) | 0.657 | 0.817 | 0.827 | **0.811** |
| physics-augmented XGBoost (exp_035) | — | — | — | 0.753 |
| physics-augmented ridge (exp_044) | — | — | — | **0.802** |

Key findings:
- TL fine-tuning *hurts* under the chemistry split (0.574 vs 0.625 no-TL):
  continuation training over-weights Batch_2's recipe distribution, which
  doesn't transfer to held-out chemistries.
- Both LightGBM variants lose to physics augmentation by 0.13–0.23.
- The per-shear ridge at 0.811 exposed the decisive insight: log-viscosity
  is near-linear in the four inputs, and linear models extrapolate outside
  the training composition hull where trees cannot.  This motivated the
  exp_044 varying-coefficient ridge student, which keeps the linear
  extrapolation *and* the any-shear capability (0.802).

---

## Physics-Augmented Pipeline

### exp_029 — First physics-augmented run (raw targets)

Raw Pa·s targets.  η(1) dominates loss.  Negative η(100) predictions
observed.  R²(raw) ≈ 0.69–0.72.  Superseded by exp_030.

### exp_030 — Log1p target (headline augmented run, random split)

**Fixed split (random 80/20, n_test=126):**

| Oracle | R²(log) | R²(raw) | RMSE |
|---|---|---|---|
| power_law | **0.815** | 0.738 | 22.6 |
| constrained_cy | 0.810 | 0.732 | 23.0 |
| cy5 | 0.812 | 0.735 | 22.8 |
| sisko | 0.809 | 0.729 | 23.2 |
| williamson | 0.813 | 0.736 | 22.9 |

Note: config specifies n_aug_points=200 (original).  Production comparisons
(LOO, batch, few-shot) use n_aug_points=30 — the aug_sweep study confirmed
no difference between 10 and 500 points in log space.

### exp_031 / exp_032 — Sample-weight ablations

No improvement from up-weighting measured points (raw or log targets).
Neither sample weight ratio nor augmentation volume is the binding
constraint.

### exp_033 — No augmentation baseline

**KEY FINDING: R²(log)=0.815 — identical to exp_030 (augmented).**

The augmentation hypothesis was falsified in log space *at the measured
shears*.  The raw-space apparent gain (0.738 vs 0.704) was a η(1) scale
artefact.  Augmentation's real contribution is structural — see the
held-out-shear study below.

### exp_034 — Optuna-tuned augmented

R²(raw) = 0.695–0.728 — no improvement over untuned exp_030.  Tuning on a
random slice of the augmented training set did not transfer.

### exp_035 — Chemistry-standardised split (strict hold-out)

**R²(log) = 0.753** (power_law oracle, n_test=132, 16 Composite_Mix_IDs
held out, all 5 oracles, n_aug_points=30).

The 0.062 gap vs exp_030's random-split 0.815 quantifies recipe-level
leakage in the random split.  This was the honest headline number until
exp_044.

### exp_038 — Full engineered feature set

Best oracle R²(log) = 0.757 (cy5) vs 0.753 baseline — the 12-feature
engineered set adds ~nothing under the chemistry split.  Tree students
already capture what those features encode.

### exp_040 — Monotone shear-thinning constraint (honest negative + trade-off)

XGBoost student constrained non-increasing in log_gamma (shear-thinning
prior; XGBoost native `monotone_constraints`).

| Oracle | R²(log) |
|---|---|
| constrained_cy | 0.735 |
| power_law | 0.735 |
| sisko | 0.725 |
| cy5 | 0.700 |
| williamson | 0.692 |

At the measured shears the constraint *costs* ~0.02 R²(log).  In
curve-shape terms (oracle pseudo-gt bands, below) it improves both
above-γ̇=10 bands by ~0.07–0.11 while *degrading* the below-γ̇=1 band by
~0.07–0.10: it removes spurious shear-thickening wiggles at high shear
but flattens genuine low-shear structure.  A trade-off, not a win.

### exp_041 / exp_042 — Maron–Pierce crowding feature

Crowding term (1 − φ/82)⁻² from Solid_Content_pct, alone (exp_041, best
0.740) and combined with the monotone prior (exp_042, best 0.716).
No improvement for the tree student — consistent with exp_038.  The
crowding *rationale* (log-linear concentration dependence) succeeds in
the linear student instead (exp_044).

### exp_043 — GPR student

| Variant | R²(log) |
|---|---|
| power_law__gpr | 0.752 |
| power_law__xgb (same run) | 0.744 |
| cy5__gpr | 0.695 |

GPR matches the XGBoost student at the measured shears and adds native
curve-level predictive variance, but does not approach the ridge student.

### exp_044 — Varying-coefficient ridge student (NEW HEADLINE)

Ridge on [composition, log γ̇, composition × log γ̇] in log1p space — a
power-law surface whose consistency and flow index are linear in
composition.  α selected by efficient LOO-CV.

| Oracle | R²(log) | R²(raw) |
|---|---|---|
| power_law | **0.802** | 0.385 |
| constrained_cy | 0.802 | 0.368 |
| sisko | 0.796 | 0.391 |
| cy5 | 0.788 | 0.358 |
| williamson | 0.785 | 0.356 |

+0.049 R²(log) over the XGBoost student on identical test rows.  Why it
works: trees predict constants outside the training composition hull —
exactly the strict-split failure mode — while log-viscosity varies
near-linearly with composition over the observed range, so a linear
student extrapolates correctly.  See also: held-out-shear, curve bands,
LOO robustness, and conformal calibration below — the ridge student wins
or ties every regime except LOO median.

---

## Held-out-shear study (the structural-capability centrepiece)

Train on two of the three measured shears, evaluate on the third.  The
power-law oracle is fitted from the two training shears only (two points
exactly determine K and n — no leakage from the held-out shear).
γ̇=10 held out tests *interpolation* between training shears; γ̇=1 and
γ̇=100 test *extrapolation* below/above them.  CSVs:
`shear_holdout_results.csv`, `shear_holdout_ridge_results.csv`.

**R²(log) at the held-out shear:**

| Held-out γ̇ | XGB no_aug | XGB aug_30 | Ridge no_aug | Ridge aug_30 |
|---|---|---|---|---|
| 1 (extrap. ↓) | −0.146 | 0.495 | 0.593 | 0.624 |
| 10 (interp.) | **−2.596** | **0.675** | 0.727 | 0.712 |
| 100 (extrap. ↑) | −2.016 | 0.449 | 0.524 | **0.793** |

KEY FINDING — the augmentation decomposition:
- The tree student has *no* training signal at unseen log_gamma values and
  collapses catastrophically without augmentation; augmentation rebuilds
  the curve between/beyond training shears.  For trees, augmentation is
  structural.
- The ridge student is linear in log_gamma by construction and predicts
  held-out shears well with **zero** augmented points.  Augmentation adds
  only at the extrapolated high shear (0.524 → 0.793).
- Therefore physics augmentation primarily *repairs a missing inductive
  bias* in students that lack one, and is a smaller, targeted aid for
  students with the right bias built in.  This is the dissertation's
  sharpest statement of what the method does.

---

## Extended Evaluation Regimes

### LOO — 67-fold Composite_Mix_ID leave-one-out

CSVs: `loo_comparison_results.csv` (XGB), `loo_ridge_results.csv` (ridge);
best oracle per fold, R²(log).

| Student | Condition | Median | Mean | Std | Min | aug>raw folds |
|---|---|---|---|---|---|---|
| XGB | raw | 0.741 | −0.422 | 3.80 | −22.3 | — |
| XGB | aug | 0.747 | 0.037 | 2.38 | −13.0 | 48/67 |
| Ridge | raw | 0.689 | 0.147 | 1.52 | −6.9 | — |
| Ridge | aug | 0.719 | 0.228 | 1.42 | −7.1 | 49/67 |

- XGB has the better median (0.747 vs 0.719): with 66 of 67 recipes in
  training, most folds are interpolation, where local tree fitting is
  sharp.
- Ridge has far better tails (mean +0.23 vs +0.04; worst fold −7 vs −13):
  the same extrapolation robustness that wins the strict hold-out.
- F1 recipes (well-sampled): median R²(log) ≈ 0.74–0.82 for both.
  F2/F3 recipes (sparse): negative R² regardless of student or
  augmentation — compositional extrapolation to undersampled formulation
  families remains the binding constraint.

### Few-shot — k measurements of the new recipe leaked into training

CSV: `fewshot_comparison_results.csv` (XGB student; medians across 31
eligible IDs).

| k | raw | aug |
|---|---|---|
| 0 (LOO baseline) | 0.665 | 0.612 |
| 1 | 0.701 | 0.745 |
| 2 | 0.785 | 0.800 |

One measurement of a new recipe lifts median R²(log) by +0.04–0.13; two
approach the fixed-split ceiling (0.815).  Even a single measurement
largely resolves the per-recipe scale (K); curve shape follows.

### Cross-batch generalisation

CSV: `batch_comparison_results.csv` (XGB student).

| Regime | Condition | R²(log) | R²(raw) | RMSE |
|---|---|---|---|---|
| Batch_1→2 (37 unseen) | raw | 0.698 | 0.805 | 14.1 |
| Batch_1→2 (37 unseen) | aug | 0.707 | 0.815 | 13.7 |
| Batch_2→1 (18 unseen) | raw | 0.617 | 0.428 | 32.5 |
| Batch_2→1 (18 unseen) | aug | 0.614 | 0.443 | 32.1 |

Augmentation provides marginal benefit in B1→B2 direction; zero in B2→B1.

### Oracle curve evaluation (pseudo-ground-truth)

**CAVEAT: oracle used as ground truth — not measured data.**  Quantifies
curve-shape recovery only.  CSVs: `oracle_curve_bands*.csv`
(power_law oracle column shown; cy5 in the CSVs).

| Shear band | XGB | XGB + monotone | Ridge |
|---|---|---|---|
| 0.1–1 (below measured) | 0.456 | 0.355 | **0.651** |
| 1–10 (measured) | 0.554 | 0.526 | **0.722** |
| 10–100 (measured) | 0.656 | 0.726 | **0.811** |
| 100–1000 (above measured) | 0.612 | 0.722 | **0.726** |

- The unconstrained tree collapses below γ̇=1 where the curve is steepest.
- The monotone prior shifts error from high-shear to low-shear bands.
- The ridge student dominates **every band** — including +0.20 below the
  measured range — because the power-law surface extrapolates log-linearly
  in log γ̇ by construction.

### GPR LOO (parameter space, old framework — exp_026)

**CAVEAT: old leave-one-group-out framework (12 Formulation×Dispersant
groups), not the 67-fold chemistry LOO.**  Mix-group LOO for K: GPR 0.691
≈ XGBoost 0.652.  Batch-LOO catastrophic for all parameters (R² ≤ 0.004)
— cross-batch generalisation of physics parameters is near-zero, matching
the viscosity-space cross-batch finding.  (A 67-fold GPR-student LOO pass
is regenerable via `python -m src.cli study --name loo_comparison_gpr`.)

---

## Full regime comparison (summary table)

XGB student except where noted; R²(log).

| Regime | raw | aug | Note |
|---|---|---|---|
| Fixed split (random) | 0.815 | 0.815 | n_test=126; recipe leakage inflates |
| Strict hold-out — XGB | — | 0.753 | n_test=132 |
| Strict hold-out — **ridge** | — | **0.802** | n_test=132 |
| LOO median (67 folds) — XGB | 0.741 | 0.747 | |
| LOO median (67 folds) — ridge | 0.689 | 0.719 | better mean/tails |
| Batch_1→2 | 0.698 | 0.707 | 37 unseen recipes |
| Batch_2→1 | 0.617 | 0.614 | 18 unseen recipes |

---

## Parameter-space experiments (exp_012, exp_013, exp_025, exp_026, exp_037)

### exp_012/013 — XGBoost parameter prediction (early expanded split)

| Target | R²(test) | Notes |
|---|---|---|
| k_consistency (K, Power Law) | 0.652 | Moderately learnable |
| n_flow (n, Power Law) | 0.311 | Barely learnable — the core bottleneck |

When measured viscosities are used as inputs alongside composition,
reconstruction R²=0.812 in log space — nearly identical to physics
augmentation.  The three measured points almost fully constrain K and n.
The hard problem is predicting n for *unseen* compositions.

### exp_025 — XGBoost vs GPR parameter prediction (interpolation regime)

Random 80/20 split on the old expanded dataset — the realistic
"chemist tweaks an existing formulation" scenario.

| Model | Mean R²(test) |
|---|---|
| GPR (Matérn 2.5 + linear, ARD, 20 restarts) | **0.559** |
| XGBoost (10 Optuna trials) | 0.456 |

GPR's advantage concentrates on flow-index (n-type) parameters
(0.445 vs 0.209 for Power Law n_flow).  Neither model achieves R² > 0.45
for n_flow even in the interpolation regime.

### exp_037 — Parameter prediction on chemistry split (extrapolation regime)

| Model | Paradigm | Interpolation (exp_025) | Extrapolation (exp_037) | Drop |
|---|---|---|---|---|
| Physics-augmented XGBoost | B | 0.815 | 0.753 | −0.062 |
| GPR (params) | A | 0.559 | 0.166 | −0.393 |
| XGBoost (params) | A | 0.456 | 0.014 | −0.442 |

KEY FINDING: Paradigm B (augmented student) degrades by 0.062 moving to
extrapolation; Paradigm A (parameter prediction) degrades by 0.39–0.44.
Paradigm A must infer K and n as latent variables from composition —
unreliable outside the training distribution.  Paradigm B learns
log-viscosity directly from per-row local fits and never attempts global
parameter inference.  (With the exp_044 ridge student, Paradigm B's
extrapolation drop shrinks further: 0.815 → 0.802.)

---

## Core dissertation finding

**n (curve shape) cannot be reliably inferred from composition for
genuinely unseen recipes; K (scale) can.**  Every evaluation regime
degrades in proportion to the compositional novelty of the test set, and
the degradation maps onto the difficulty of inferring n.

Within that constraint, two design choices recover most of the available
performance:

1. **Physics augmentation** rebuilds the shear-direction signal a tree
   student lacks (held-out-shear: −2.6 → +0.68), and reduces worst-case
   LOO failures.  It does not add information at the measured shears
   (exp_030 ≡ exp_033).
2. **A student with the right inductive bias** (the varying-coefficient
   ridge: log-linear in composition and log γ̇) extrapolates where trees
   cannot, lifting the strict hold-out headline from 0.753 to **0.802**,
   dominating every curve-shape band, and yielding near-nominal conformal
   coverage.

Solving the remaining gap requires data: even one measurement of a new
recipe (few-shot k=1) is worth more than any modelling change tested.

---

## Uncertainty Quantification

### Conformal prediction (chemistry split, power_law oracle, n_aug=30)

Split-conformal on a 20% group-aware calibration hold-out of training
recipes; coverage on the 16 held-out test recipes (132 points → 44 per
shear).  CSVs: `conformal_{xgb,ridge}[_mondrian]_results.csv`.

| Student | Calibration | Nominal 90% | Nominal 95% | Mean width (log) @90% |
|---|---|---|---|---|
| XGB | marginal | 82.6% | 87.1% | 1.369 |
| XGB | Mondrian per-shear | 77.3% | 81.8% | 1.199 |
| Ridge | marginal | 88.6% | 89.4% | 1.593 |
| Ridge | Mondrian per-shear | 84.8% | **95.5%** | 1.208 |

- Marginal intervals under-cover for the XGB student: the OOD chemistry
  split violates exchangeability, consistent with its elevated test error.
- The ridge student is close to nominal (88.6% vs 90%) — better OOD
  calibration tracks its better OOD accuracy.
- Mondrian (per-shear) calibration exposes the error asymmetry the
  marginal radius hides: q(γ̇=1) ≈ 0.93–0.97 vs q(γ̇=100) ≈ 0.30–0.32 in
  log1p units — the low-shear viscosity is ~3× harder.  With only 20
  calibration scores per shear the Mondrian quantiles are noisy; at the
  95% level the ridge+Mondrian combination achieves nominal coverage
  (95.5%) at a narrower mean width than the marginal interval.

### GPR calibration and uncertainty analysis (gpr_uq study)

Source: exp_037 GPR test-set predictions (16 recipes × 5 models × 11 targets).

| Nominal | Empirical (mean over models/targets) |
|---|---|
| 68% | 50.1% |
| 95% | 80.9% |

GPR intervals are systematically overconfident on the OOD test set.

**σ as a difficulty proxy (n_flow, power law):** Spearman ρ(σ, |error|)
= +0.468 (p = 0.068, n = 16).  Directionally correct but the σ range is
compressed (0.079–0.090) — all OOD recipes look equally novel to the
kernel.  σ-guided selection preferentially picks harder recipes first
(early gain −0.020 MAE), supporting measure-first prioritisation as a
weak but real signal; a composition-space diversity criterion would
likely be stronger.

**Overall UQ conclusion:** calibration guarantees derived from the
training distribution do not transfer to novel chemistries.  Of the
options tested, the ridge student with Mondrian per-shear conformal
calibration is the most trustworthy uncertainty stack, and its intervals
are honest about the low-shear difficulty.
