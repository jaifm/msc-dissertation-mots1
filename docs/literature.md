# Literature Notes — Methodological Ideas and Their Academic Basis

Every non-obvious methodological choice in this project is recorded here
with the literature that motivates it, what was implemented, and what the
experiments found.  Use this file as the seed for the dissertation's
related-work and methodology chapters.

> **Citation hygiene:** the references below were collected during
> development; verify page numbers, volumes and DOIs against the original
> sources before the final submission.

---

## 1. Physics-augmented training (Approach 3 — the core method)

**Idea.** Use a fitted rheological model not as a regression target but as
a *data-augmentation oracle*: fit each training row's three measured
viscosities, sample the fitted curve at N log-uniform shear rates, and
train a single student `f(composition, log γ̇) → η` on the expanded set.
Test rows are never fitted or augmented (the leakage contract in
`src/physics_augmented/augment.py`).

**Literature.** This is an instance of *theory-guided / physics-informed
machine learning*: domain models supply inductive bias or synthetic data
where measurements are sparse.

- Karpatne, A., Atluri, G., Faghmous, J., et al. (2017). *Theory-guided
  data science: A new paradigm for scientific discovery from data.*
  IEEE Transactions on Knowledge and Data Engineering 29(10), 2318–2337.
- Willard, J., Jia, X., Xu, S., Steinbach, M., Kumar, V. (2022).
  *Integrating scientific knowledge with machine learning for engineering
  and environmental systems.* ACM Computing Surveys 55(4), 1–37.
- von Rueden, L., Mayer, S., Beckh, K., et al. (2021). *Informed machine
  learning — a taxonomy and survey of integrating prior knowledge into
  learning systems.* IEEE TKDE 35(1), 614–633.

**Findings.** Augmentation does not raise mean accuracy at the measured
shears (exp_030 vs exp_033: identical R²(log)=0.815) but is *structural*
for shear-rate generalisation with a tree student: the held-out-shear
study shows R²(log) at an unseen shear collapsing to −2.6 without
augmentation and recovering to +0.68 with it.  See §4 for the learner-
dependence of this conclusion.

## 2. Monotone shear-thinning constraint (exp_040)

**Idea.** Battery cathode slurries are shear-thinning across the
industrial 0.1–1000 s⁻¹ window, so predicted viscosity should be
non-increasing in shear rate.  XGBoost supports per-feature monotone
constraints natively; we constrain the encoded `log_gamma` column to −1.

**Literature.**
- Karpatne et al. (2017), as above — constraints as theory-guided bias.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting
  system.* KDD '16, 785–794.  (Implementation basis; monotone constraints
  are a documented feature of the library.)
- Shear-thinning behaviour of NMC cathode slurries: Bauer, W. &
  Nötzel, D. (2014). *Rheological properties and stability of NMP based
  cathode slurries for lithium ion batteries.* Ceramics International
  40(3), 4591–4598.

**Findings (honest negatives included).** At the measured shears the
constraint slightly *reduces* pooled R²(log) (0.753 → 0.735).  In
curve-shape terms (oracle pseudo-ground-truth) it improves the
above-measured bands substantially (10–100: 0.656 → 0.726; 100–1000:
0.612 → 0.722) but *worsens* the below-measured band (0.456 → 0.355):
the constraint removes spurious shear-thickening wiggles at high shear
while flattening genuine low-shear structure.  Reported as a trade-off,
not an unconditional win.

## 3. Maron–Pierce crowding feature (exp_041)

**Idea.** Relative viscosity of concentrated suspensions diverges as the
solids loading approaches a maximum packing fraction.  The raw
`Solid_Content_pct` column spans only 69–77 wt%, so the divergence is
invisible to a tree model; we pre-compute `(1 − φ/φ_max)⁻²` with
φ_max = 82 wt% as an explicit feature.

**Literature.**
- Krieger, I.M. & Dougherty, T.J. (1959). *A mechanism for non-Newtonian
  flow in suspensions of rigid spheres.* Transactions of the Society of
  Rheology 3, 137–152.
- Maron, S.H. & Pierce, P.E. (1956). *Application of Ree-Eyring
  generalized flow theory to suspensions of spherical particles.*
  Journal of Colloid Science 11(1), 80–95.
- Mooney, M. (1951). *The viscosity of a concentrated suspension of
  spherical particles.* Journal of Colloid Science 6(2), 162–170.

**Findings.** No improvement under the chemistry split (best oracle
0.740 vs 0.753 baseline); combined with the monotone constraint it is
slightly worse still (0.716).  Consistent with exp_038's finding that
engineered composition features add little for the tree student.  The
crowding *rationale* resurfaces successfully in the linear student (§4),
where log-linear concentration dependence is the operative bias.

## 4. Varying-coefficient ridge student (exp_044 — new headline model)

**Idea.** Trees predict constants outside the training hull, which is
exactly the failure mode of a strict unseen-chemistry split.  Log-
viscosity varies smoothly — nearly linearly — with composition over the
observed range (exponential concentration laws of the Mooney type), so a
linear model in log space extrapolates where trees cannot.  The student
is ridge regression on `[x, log γ̇, x · log γ̇]`: with the log1p target
this is an Ostwald–de Waele power-law surface whose consistency *and*
flow index are linear functions of composition — a varying-coefficient
model.

**Literature.**
- Hastie, T. & Tibshirani, R. (1993). *Varying-coefficient models.*
  Journal of the Royal Statistical Society, Series B 55(4), 757–796.
- Ostwald–de Waele power law: de Waele, A. (1923), Oil and Color Chem.
  Assoc. Journal 6, 33–88; Ostwald, W. (1925), Kolloid-Zeitschrift 36,
  99–117.
- Mooney (1951), as above — exponential/log-linear concentration
  dependence motivating linearity in log space.
- Golub, G.H., Heath, M., Wahba, G. (1979). *Generalized cross-validation
  as a method for choosing a good ridge parameter.* Technometrics 21(2),
  215–223.  (RidgeCV's efficient leave-one-out α selection.)
- On tree extrapolation limits: Zhang, H., Nettleton, D., Zhu, Z. (2019).
  *Regression-enhanced random forests.* arXiv:1904.10416.

**Findings.** The strongest single improvement of the project:
- Strict chemistry hold-out pooled R²(log): **0.802** vs 0.753 (XGBoost
  student), 0.752 (GPR student), 0.574 (transfer-learning LightGBM).
- Held-out-shear study: the ridge student predicts an unseen shear well
  *without any augmentation* (R²(log) 0.59/0.73/0.52 at γ̇ = 1/10/100)
  where XGBoost collapses (−0.15/−2.60/−2.02).  Augmentation still adds
  at the extrapolated high shear (0.52 → 0.79).  This decomposes the
  augmentation effect: it mainly *repairs the tree student's missing
  inductive bias along log γ̇*; a student with the right bias needs far
  less of it.
- Curve-shape (oracle pseudo-gt) R² improves in every band, including
  0.456 → 0.651 below the measured range.

## 5. Conformal prediction and Mondrian (per-shear) calibration

**Idea.** Distribution-free prediction intervals via split-conformal
calibration on held-out training recipes; extended with Mondrian
(group-conditional) calibration per shear rate, since the error
distribution at η(1) is ~3× wider than at η(100) and a single marginal
radius both over-covers easy groups and under-covers hard ones.

**Literature.**
- Vovk, V., Gammerman, A., Shafer, G. (2005). *Algorithmic Learning in a
  Random World.* Springer.  (Conformal framework; Mondrian variant.)
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J., Wasserman, L.
  (2018). *Distribution-free predictive inference for regression.*
  Journal of the American Statistical Association 113(523), 1094–1111.
- Boström, H. & Johansson, U. (2020). *Mondrian conformal regressors.*
  Proceedings of COPA 2020, PMLR 128, 114–133.
- Romano, Y., Patterson, E., Candès, E. (2019). *Conformalized quantile
  regression.* NeurIPS 32.  (Adaptive-width alternative; future work.)

**Findings.** Under the OOD chemistry split the XGBoost student's
marginal intervals under-cover (82.6% at 90% nominal) — exchangeability
is violated by design.  The ridge student is markedly better calibrated
(88.6% at 90% nominal; ridge + Mondrian reaches 95.5% at 95% nominal),
consistent with its smaller OOD error.  Mondrian radii expose the
per-shear error asymmetry (q ≈ 0.93–0.97 at γ̇=1 vs 0.30–0.32 at γ̇=100).

## 6. GPR students and uncertainty quantification (exp_037, exp_043, gpr_uq)

**Idea.** Gaussian-process regression supplies posterior predictive
variance — usable as an acquisition signal for deciding which new recipe
to measure first (active learning).

**Literature.**
- Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for
  Machine Learning.* MIT Press.
- Settles, B. (2009). *Active Learning Literature Survey.* University of
  Wisconsin–Madison, CS Technical Report 1648.

**Findings.** As a PA student GPR matches XGBoost (0.752 vs 0.753) but
not ridge.  Parameter-space GPR intervals are overconfident OOD (50.1%
empirical at 68% nominal); σ correlates with per-recipe error for the
bottleneck parameter (Spearman ρ = 0.47, p = 0.068, n = 16), supporting
σ-guided measure-first prioritisation as a weak but directionally
correct signal.

## 7. Transfer-learning replication and audit (Approach 2)

**Idea.** Faithful replication of the paper's LightGBM continuation
training, then a leakage audit, then (new) a transplant onto the
chemistry split so TL and physics augmentation are compared on identical
held-out rows.

**Literature.**
- El-Dalahmeh, M., et al. (2025). [TL for battery slurry viscosity.]
  Energy and AI 22, 100661.
- Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A highly
  efficient gradient boosting decision tree.* NeurIPS 30.
- Pan, S.J. & Yang, Q. (2010). *A survey on transfer learning.* IEEE
  TKDE 22(10), 1345–1359.  (Framing of continuation training as
  parameter-transfer.)

**Findings.** Under the chemistry split, TL LightGBM reaches pooled
R²(log) 0.574 (no-TL pooled LightGBM 0.625) vs physics-augmented 0.753
(XGBoost) / 0.802 (ridge) on the *same test rows*.  Two structural
deficits: γ̇ is not an input (three shear-blind models; no prediction
between/beyond measured shears), and the 4-input tree model inherits the
extrapolation failure.  The paper's claimed R² ≈ 0.85–0.92 remained
unreproducible under every chaining interpretation (see RESULTS.md).

## 8. Rheological model forms (oracles)

- Power law: Ostwald (1925) / de Waele (1923), as above.
- Carreau, P.J. (1972). *Rheological equations from molecular network
  theories.* Transactions of the Society of Rheology 16(1), 99–127.
- Yasuda, K., Armstrong, R.C., Cohen, R.E. (1981). *Shear flow properties
  of concentrated solutions of linear and star branched polystyrenes.*
  Rheologica Acta 20(2), 163–178.  (Carreau–Yasuda / CY5.)
- Cross, M.M. (1965). *Rheology of non-Newtonian fluids: a new flow
  equation for pseudoplastic systems.* Journal of Colloid Science 20(5),
  417–437.
- Williamson, R.V. (1929). *The flow of pseudoplastic materials.*
  Industrial & Engineering Chemistry 21(11), 1108–1111.
- Sisko, A.W. (1958). *The flow of lubricating greases.* Industrial &
  Engineering Chemistry 50(12), 1789–1792.

## 9. Domain context — cathode slurry processing

- Hawley, W.B. & Li, J. (2019). *Electrode manufacturing for lithium-ion
  batteries — analysis of current and next generation processing.*
  Journal of Energy Storage 25, 100862.
- Bauer & Nötzel (2014), as above.
- Kraytsberg, A. & Ein-Eli, Y. (2016). *Conveying advanced Li-ion battery
  materials into practice: the impact of electrode slurry preparation
  skills.* Advanced Energy Materials 6(21), 1600655.

## 10. Retired direction — viscosity-space loss (do not revisit)

A custom XGBoost objective scoring predicted physics parameters against
the three measured viscosities (JAX-differentiated) was prototyped and
retired: the loss only sees the measured shears, so nothing pushes the
predicted parameters to be sensible elsewhere on the curve — directly
opposing the any-shear deployment goal.  Kept here so the idea is not
re-proposed.
