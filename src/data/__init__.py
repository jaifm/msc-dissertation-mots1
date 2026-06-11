"""Data subpackage: raw data preparation and split loading.

Modules
-------
prepare.py
    Combines raw batch CSVs, expands formulation labels into numeric
    component percentages, and creates a reproducible group-aware
    train/test split.  Run once as a script before any experiments.

splits.py
    Loads the pre-computed split CSVs and validates the no-leakage
    invariant (no Composite_Mix_ID appears in both train and test).
"""
