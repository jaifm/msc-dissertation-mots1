# Composition → Rheology Prediction for Battery Slurries

MSc dissertation project: predict the full viscosity–shear-rate curve of a
lithium-ion cathode slurry from its composition alone, before any wet-lab
measurement.  The model must be queryable at *any* shear rate in the
industrial 0.1–1000 s⁻¹ window — not just the three rates at which the
training data was measured.

The methodological story (three approaches, what failed, and why the
current physics-augmented + varying-coefficient-ridge approach wins) is in
[docs/methodology.md](docs/methodology.md).  Headline numbers live in
[RESULTS.md](RESULTS.md); regeneration instructions in
[docs/reproducibility.md](docs/reproducibility.md); literature backing for
every methodological idea in [docs/literature.md](docs/literature.md).

## Quick start

```bash
conda env create -f environment.yml
conda activate msc-dissertation-mots1-v2
pip install lightgbm        # transfer-learning replication only
pytest tests/ -q

# run one experiment
python -m src.cli run --config exp_044_pa_ridge_student

# run an evaluation study (multi-run campaign)
python -m src.cli study --name shear_holdout
```

## Layout

```
configs/
  base.yaml                  defaults; every experiment inherits via `defaults: base`
  experiments/               one YAML per experiment (exp_001 … exp_044)
  studies/                   one YAML per evaluation study
data/processed/
  splits/                    legacy random splits (exp_001–028)
  splits_chemistry/          fixed hash-pinned chemistry splits + MANIFEST.json
  xgboost_experiments/       run dirs (gitignored) + manifest + study summary CSVs
docs/                        methodology, reproducibility, literature notes
notebooks/                   thin interactive drivers
scripts/
  prepare_chemistry_splits.py  the ONLY writer of splits_chemistry/
  run_all_experiments.py       batch orchestrator
src/
  config.py                  typed-dataclass YAML loader + validation
  data/, physics/, features.py, pipeline.py, tuning.py, gpr.py,
  curve_metrics.py, shap_analysis.py, viz.py
  experiments/               section runners (Approach 1: parameter prediction)
  transfer_learning/         Approach 2: El-Dalahmeh et al. replication + audit
  physics_augmented/         Approach 3: oracle-augmented students (xgb/gpr/ridge)
  studies/                   config-driven evaluation campaigns
  cli.py                     run / list / study / list-studies
tests/                       config, pipeline, studies, physics-prior tests
```

## The two execution concepts

**Experiment** — a single train/evaluate cycle on one split, defined
entirely by a YAML in `configs/experiments/`.  Adding an experiment means
adding a file; never editing code.  Every run gets a unique id, never
overwrites, and snapshots its merged config next to its outputs.

**Study** — an evaluation campaign orchestrating many runs (67-fold LOO,
few-shot recovery, cross-batch, held-out-shear, conformal intervals, UQ,
dense curve evaluation).  Each study module documents its parameters in a
`DEFAULTS` dict, overridden by `configs/studies/<name>.yaml`.  Summary
CSVs land in `data/processed/xgboost_experiments/` and are tracked in git.

## Evaluation contract

All headline claims use the fixed chemistry-based splits
(`splits_chemistry/`, SHA-256-pinned in its MANIFEST.json): a chemistry is
a unique `Composite_Mix_ID` (full recipe), and test chemistries never
appear in training at any level.  The primary metric is **R²(log)**
(log1p space) — raw-space R² is dominated by the η(1) scale.  Physics
augmentation is applied to training rows only (the leakage contract in
`src/physics_augmented/augment.py`).

## Adding a new methodology

Build a subpackage under `src/`, expose a `run_<name>` entry point,
register the section key in `_VALID_SECTIONS` (`src/config.py`), add a
lazy-import dispatch in `src/experiments/runner.py`, and give any new
config knobs defaults so older YAMLs still parse.
`src/transfer_learning/` and `src/physics_augmented/` are the templates.
