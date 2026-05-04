"""Top-level experiment orchestration: config in → results out.

OVERVIEW
--------
The runner is the single public entry point for executing an experiment.
Given a loaded ``Config`` object, it:

  1. Loads the pre-computed train/test splits from disk.
  2. Fits the configured physics models to every row in both splits, producing
     per-row rheology parameter datasets.
  3. Dispatches to each requested section (baseline, sweep, LOO) in order.
  4. Optionally persists all results, config snapshot, and manifest entry.
  5. Prints a compact summary to stdout.

SECTION DISPATCH
----------------
Which sections run is controlled by ``cfg.run.sections``.  The three sections
are independent and can be run in any combination:

  ``baseline``  -- Full model comparison with Optuna tuning and collected
                   predictions.  The primary section for ranking models and
                   generating plots.

  ``sweep``     -- Runs the baseline twice to compare standard vs log-
                   transformed numeric features.  Useful for validating the
                   feature engineering decisions.

  ``loo``       -- Leave-one-out evaluation for generalisability assessment.
                   Slower than the other sections because it refits the model
                   once per fold.

DESIGN PRINCIPLES
-----------------
The runner is intentionally thin — it contains no model logic, only
orchestration.  Each physics fitter, preprocessor, tuning loop, and
evaluation metric lives in its own module, so the runner's job is purely
to connect those pieces in the right order and collect their outputs.

All outputs are returned as a typed ``ExperimentResults`` object.  Saving
to disk is optional (``persist=False`` is useful for notebook iteration
without cluttering the results directory).
"""
from __future__ import annotations

import time
from typing import Dict

import pandas as pd

from ..config import Config
from ..data.splits import load_splits
from ..physics.datasets import build_all_physics_datasets
from . import loo as loo_module
from . import sweep as sweep_module
from .results import ExperimentResults, save


def run_experiment(cfg: Config, *, persist: bool = True) -> ExperimentResults:
    """Execute every section in ``cfg.run.sections`` and return the results.

    Parameters
    ----------
    cfg:
        Fully loaded and validated experiment configuration.
    persist:
        When True (the default), write all results to ``cfg.results_dir``
        and update the shared experiment manifest.  Set to False for quick
        interactive runs where disk output is not needed.

    Returns
    -------
    ExperimentResults
        Contains one ``SectionResult`` for each section that was run, plus
        the experiment ID and creation timestamp.
    """
    print("="*72)
    print(f"ACTIVE EXPERIMENT: {cfg.id}")
    print("="*72)
    print(f"Description : {cfg.description}")
    print(f"Hypothesis  : {cfg.hypothesis}")
    print(f"Sections    : {cfg.run.sections}")
    print(f"Trials      : {cfg.xgb.n_trials}")
    print(f"Engineered  : {cfg.features.use_engineered}")
    print(f"Log numeric : {cfg.features.log_numeric}")
    print(f"Seed        : {cfg.seed}")

    # Step 1: load splits and validate no-leakage invariant.
    train_df, test_df = load_splits(cfg)
    print(f"\nLoaded splits: train={len(train_df)} rows, test={len(test_df)} rows")

    # Step 2: fit physics models to every row.  This is done once and reused
    # by all sections, so the per-row fitting cost is not repeated per section.
    physics_datasets = build_all_physics_datasets(train_df, test_df, cfg)
    print(f"Built physics datasets for {len(physics_datasets)} models.")

    results = ExperimentResults(experiment_id=cfg.id)

    # Step 3: dispatch to each requested section, timing the full run.
    _start = time.perf_counter()
    if "baseline" in cfg.run.sections:
        results.add(sweep_module.run_baseline(physics_datasets, cfg))
    if "sweep" in cfg.run.sections:
        results.add(sweep_module.run_sweep(physics_datasets, cfg))
    if "loo" in cfg.run.sections:
        results.add(loo_module.run_loo(physics_datasets, cfg))
    if "gpr_baseline" in cfg.run.sections:
        from . import gpr_section as gpr_module
        results.add(gpr_module.run_gpr_baseline(physics_datasets, cfg))
    if "gpr_xgb_ensemble" in cfg.run.sections:
        from . import ensemble_section as ens_module
        results.add(ens_module.run_gpr_xgb_ensemble(physics_datasets, cfg))
    results.duration_seconds = round(time.perf_counter() - _start, 1)

    # Step 4: optionally persist all outputs.
    if persist:
        out_dir = save(cfg, results)
        print(f"\nRun ID   : {results.run_id}")
        print(f"Results  : {out_dir}")
        print(f"Duration : {_fmt_duration(results.duration_seconds)}")

    # Step 5: print a compact best-model summary for quick inspection.
    best = (
        results.best("baseline") or results.best("sweep")
        or results.best("loo") or results.best("gpr_baseline")
        or results.best("gpr_xgb_ensemble")
    )
    if best:
        print(
            f"\nBest model (mean test R²): {best['model_name']} "
            f"— R²={best['test_r2']:.3f}, RMSE={best['test_rmse']:.3f}"
        )
    return results


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"
