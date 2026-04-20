"""Load the pre-computed train/test split files and enforce no-leakage invariants.

OVERVIEW
--------
The train/test split is generated once by ``data/prepare.py`` and stored as
two CSV files on disk.  This module loads those files and validates that the
group-level isolation guarantee is still intact: no value of
``cfg.data.group_column`` (Composite_Mix_ID) should appear in both splits.

WHY A SEPARATE LOADER?
-----------------------
Keeping the split fixed on disk (rather than re-splitting at runtime) ensures
that:
  1. Results from different experiments are comparable — they all use the same
     train/test boundary.
  2. The split is version-controlled as a data artefact, so it can be audited
     independently of the code that produced it.
  3. The overlap check runs every time the data is loaded, so any accidental
     modification of the CSV files (e.g. manual editing that moves a row) is
     caught early rather than silently producing invalid evaluation metrics.

The loader raises immediately on failure rather than returning an empty
DataFrame or silently ignoring the problem, because a bad split would corrupt
every metric reported by the experiment.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from ..config import Config


def load_splits(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) after validating the no-leakage invariant.

    Parameters
    ----------
    cfg:
        Experiment configuration supplying paths and the group column name.

    Returns
    -------
    (train_df, test_df)
        Raw DataFrames exactly as stored in the CSV files, with no further
        processing.  Downstream code (physics fitting, feature engineering)
        is responsible for any transformations.

    Raises
    ------
    FileNotFoundError
        If either split file does not exist at the path derived from ``cfg``.
    ValueError
        If the group column is absent from either DataFrame.
    RuntimeError
        If any group ID appears in both the train and test splits, indicating
        that the CSV files have been corrupted or incorrectly regenerated.
    """
    if not cfg.train_path.exists():
        raise FileNotFoundError(f"Train split not found at {cfg.train_path}")
    if not cfg.test_path.exists():
        raise FileNotFoundError(f"Test split not found at {cfg.test_path}")

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    gcol = cfg.data.group_column
    if gcol not in train_df.columns or gcol not in test_df.columns:
        raise ValueError(f"Group column {gcol!r} missing from splits")

    # The overlap check is O(n) via set intersection and catches any case where
    # a group ended up in both splits — which would make the test set non-
    # independent and inflate all reported R² values.
    overlap = set(train_df[gcol].unique()) & set(test_df[gcol].unique())
    if overlap:
        raise RuntimeError(
            f"Split leakage: {len(overlap)} groups appear in both splits "
            f"(examples: {sorted(overlap)[:3]})"
        )
    return train_df, test_df
