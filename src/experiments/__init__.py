"""Experiments subpackage: section runners, results containers, and persistence.

Modules
-------
runner.py
    Top-level orchestration.  ``run_experiment(cfg)`` is the single entry
    point used by notebooks and the CLI.

sweep.py
    Baseline model-comparison sweep and the log-numeric-feature variant.
    ``run_baseline`` and ``run_sweep`` are the two public entry points.

loo.py
    Leave-one-out generalisability evaluation.  ``run_loo`` runs row-level
    and mix-level LOO folds with fixed XGBoost hyperparameters.

results.py
    ``SectionResult`` and ``ExperimentResults`` dataclasses, plus ``save``
    for writing CSVs, config snapshots, and the experiment manifest.
    ``annotate_run`` adds free-text notes to completed runs in the manifest.
"""
