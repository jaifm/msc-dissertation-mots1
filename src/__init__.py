"""msc-dissertation-project-v2 — battery slurry rheology prediction.

Top-level package exposing the public API used by notebooks and the CLI.
All modelling logic lives in submodules; import from those directly for
access to individual classes and functions.

Package layout
--------------
src/
  paths.py            -- project-root resolution (PROJECT_ROOT, CONFIGS_DIR)
  config.py           -- YAML config loading and the Config dataclass
  features.py         -- composition feature engineering
  pipeline.py         -- sklearn preprocessing pipeline (impute + OHE)
  tuning.py           -- XGBoost Optuna tuning and evaluation
  cli.py              -- command-line entry point
  viz.py              -- result plotting functions
  data/
    prepare.py        -- raw CSV cleaning, formulation expansion, splitting
    splits.py         -- load pre-computed train/test split CSVs
  physics/
    models.py         -- rheology model fitters (CY5, Power Law, etc.)
    datasets.py       -- per-row physics fitting to produce target columns
  experiments/
    runner.py         -- top-level orchestration (config → results)
    sweep.py          -- baseline and log-numeric-feature variant sweep
    loo.py            -- leave-one-out generalisability evaluation
    results.py        -- result containers and disk persistence
"""
