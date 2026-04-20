# msc-dissertation-project-v2

Config-first rebuild of the XGBoost rheology pipeline. Every experiment is
defined by a YAML file under `configs/experiments/`; the notebook is a thin
orchestration layer that loads a config and delegates to `src/`.

## Layout

```
configs/
  base.yaml                      defaults (seed, paths, composition cols, targets)
  experiments/
    exp_001_baseline.yaml
    exp_002_trials_20.yaml
    exp_003_raw_only.yaml
    exp_004_trials_20_raw.yaml
    exp_010_loo_only.yaml

src/
  config.py                      load_config(name) -> Config; deep-merge + validation
  paths.py                       project-root-relative path resolution
  data/
    prepare.py                   clean + expand + create_reproducible_group_split
    splits.py                    load train/test CSVs + no-leakage assertion
  physics/
    models.py                    fit_cy5 / power_law / sisko / williamson / constrained_cy
    datasets.py                  build_physics_dataset(df, model_name, cfg)
  features.py                    engineer_composition_features()
  pipeline.py                    make_preprocessor() + r2_mse_rmse()
  tuning.py                      tune_and_evaluate_xgb() + Optuna objective
  experiments/
    sweep.py                     sweep_all_models(cfg, ...) -> ExperimentResults
    loo.py                       run_loo(cfg, ...) -> ExperimentResults
    results.py                   ExperimentResults + save(cfg, results)
    runner.py                    run_experiment(cfg) orchestrator
  viz.py                         charts 1-8 as standalone functions

notebooks/
  01_run_experiment.ipynb        pick config -> runner -> save -> plots

tests/
  test_config.py                 all YAMLs parse + validate
  test_splits.py                 no leakage, seed-deterministic
```

## Quick start

```python
from src.config import load_config
from src.experiments.runner import run_experiment

cfg = load_config("exp_001_baseline")
results = run_experiment(cfg)   # runs the sections listed in cfg.run.sections
```

Or from the CLI:

```bash
python -m src.cli run --config exp_001_baseline
```

## Adding a new experiment

Create a new YAML in `configs/experiments/`. Inherit from `base` and override
only what changes:

```yaml
defaults: base
id: exp_005_target_log1p
description: "Log1p transform on non-negative targets"
hypothesis: "Stabilises long-tailed targets (eta_0) and lifts their R²."
xgb:
  target_transform: log1p_nonnegative
run:
  sections: [baseline]
```

Then set `ACTIVE = "exp_005_target_log1p"` at the top of the runner notebook
and run. No code changes required.

## Notes on this rebuild

- The experimental feel of the original notebook is preserved (you still open a
  notebook, hit Run All, see plots). What's different is that the
  implementation lives in `src/` and the *variation* lives in YAML.
- Paths in `base.yaml` point at the existing processed data in
  `../msc-dissertation-project-MOTS1/data/processed/splits/` so no data copy is
  required. Change the `paths.splits` key to point elsewhere if you want to
  isolate runs.
- Results are written to `data/processed/xgboost_experiments/<id>/` inside this
  project, keeping v2 runs separate from the original project's outputs.
