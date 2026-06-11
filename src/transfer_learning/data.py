"""Batch-aware data loading and the paper's four-input feature subset.

The El-Dalahmeh et al. study uses four input features (Formulation,
Dispersant_Type, Solid_Content_pct, Solid_Dispersant_pct) and three viscosity
targets (η at γ̇ ∈ {1, 10, 100} 1/s).  This module loads those inputs from
the project's processed splits and partitions the data by Source_Batch.

WHY THIS LIVES OUTSIDE src/data/
--------------------------------
``src/data/splits.py`` loads the project's standard group-aware splits.  The
paper's protocol uses a different split (by Source_Batch, optionally with a
row-level shuffle within Batch_2), so a shared loader would have to grow
configuration that the rest of the pipeline doesn't need.  Keeping the
batch-aware loader inside ``transfer_learning`` preserves the isolation
contract — the main pipeline keeps its existing splitting logic unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config


# ---------------------------------------------------------------------------
# Paper's input / target columns
# ---------------------------------------------------------------------------

# Column names for the four inputs the paper uses.
PAPER_INPUT_COLS: List[str] = [
    "Formulation",
    "Dispersent_Type",
    "Solid_Content_pct",
    "Solid_Dispersant_pct",
]

# Viscosity target columns expected in the paper protocol.
PAPER_VISCOSITY_COLS: List[str] = [
    "Viscosity_at_shear_rate_1_1/s",
    "Viscosity_at_shear_rate_10_1/s",
    "Viscosity_at_shear_rate_100_1/s",
]


@dataclass
class TlSplit:
    """A batch-partitioned dataset prepared for the TL pipeline.

    Attributes
    ----------
    pretrain:
        Source-batch dataframe (Batch_1 by default).  All of it is used to
        pre-train the source-batch model.
    target:
        Target-batch dataframe (Batch_2 by default).  An 80/20 split is
        applied inside this set during evaluation.
    feature_cols:
        Column names of the four input features after Formulation derivation.
        Categorical columns are kept as plain object/string dtypes; one-hot
        encoding is applied at the learner boundary so different learners
        can use different encoding schemes if needed.
    target_cols:
        Column names of the three viscosity targets.
    group_col:
        ``Composite_Mix_ID`` — the canonical group label, retained so the
        leakage-audit variant can use it for group-aware splitting.
    """
    pretrain: pd.DataFrame
    target: pd.DataFrame
    feature_cols: List[str]
    target_cols: List[str]
    group_col: str


# ---------------------------------------------------------------------------
# Loading and batch partitioning
# ---------------------------------------------------------------------------


def load_full_dataset(cfg: Config) -> pd.DataFrame:
    """Concatenate the project's train and test splits into a single frame.

    The TL pipeline needs to re-split by Source_Batch rather than reuse the
    main pipeline's group-aware split, so it loads both splits and joins them
    back together.  The resulting DataFrame has the same schema as the
    individual split files — no columns added or dropped here.
    """
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    full = pd.concat([train, test], axis=0, ignore_index=True)
    return full


def prepare_tl_split(
    cfg: Config,
    *,
    source_batch: str = "Batch_1",
    target_batch: str = "Batch_2",
    solid_dispersant_col: str = "Solid_Additive_pct",
) -> TlSplit:
    """Load the master dataset and partition it by Source_Batch.

    Parameters
    ----------
    cfg:
        Experiment config — used only for its ``train_path`` and ``test_path``.
    source_batch, target_batch:
        Values of the ``Source_Batch`` column that select the pretrain and
        target subsets respectively.  Defaults match the paper's setup.
    solid_dispersant_col:
        Name of the column in the project's CSVs that corresponds to the
        paper's ``Solid_Dispersant_pct``.  The project uses
        ``Solid_Additive_pct``; this argument lets the mapping be overridden
        in case the schema changes.

    Returns
    -------
    TlSplit with the pretrain/target frames and column metadata.
    """
    full = load_full_dataset(cfg)

    if "Source_Batch" not in full.columns:
        raise KeyError("Source_Batch column not found in master dataset")
    if "Formulation" not in full.columns:
        raise KeyError("Formulation column not found — re-run src.data.prepare to regenerate splits")

    full = full.copy()
    if solid_dispersant_col in full.columns and "Solid_Dispersant_pct" not in full.columns:
        full["Solid_Dispersant_pct"] = full[solid_dispersant_col]

    pretrain = full[full["Source_Batch"] == source_batch].reset_index(drop=True)
    target = full[full["Source_Batch"] == target_batch].reset_index(drop=True)

    if pretrain.empty:
        raise ValueError(
            f"No rows in pretrain batch {source_batch!r}.  "
            f"Available batches: {sorted(full['Source_Batch'].unique().tolist())}"
        )
    if target.empty:
        raise ValueError(
            f"No rows in target batch {target_batch!r}.  "
            f"Available batches: {sorted(full['Source_Batch'].unique().tolist())}"
        )

    return TlSplit(
        pretrain=pretrain,
        target=target,
        feature_cols=list(PAPER_INPUT_COLS),
        target_cols=list(PAPER_VISCOSITY_COLS),
        group_col="Composite_Mix_ID",
    )


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def one_hot_encode_inputs(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    fit_categories: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """One-hot encode the categorical inputs in the paper's feature set.

    ``Formulation`` and ``Dispersent_Type`` are categorical; the remaining
    inputs are numeric and pass through unchanged.  Categories from a fitted
    training set can be passed via ``fit_categories`` so the test set's encoded
    columns line up exactly with the training set's, even when the test set
    doesn't contain every category.

    Parameters
    ----------
    df:
        DataFrame containing every column in ``feature_cols``.
    feature_cols:
        Order-preserving list of column names to encode.
    fit_categories:
        Optional dict ``{column_name: [category_values_in_order]}``.  When
        supplied, every test-set column is reindexed against the training
        categories so dropped/added categories don't break alignment.

    Returns
    -------
    (encoded_df, fit_categories_used) — the second element can be passed to
    a subsequent call so test-set encodings match the training set's columns.
    """
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    fit_categories = dict(fit_categories) if fit_categories else {}
    encoded_pieces: List[pd.DataFrame] = []
    if num_cols:
        encoded_pieces.append(df[num_cols].astype(float).reset_index(drop=True))
    for c in cat_cols:
        if c in fit_categories:
            cats = fit_categories[c]
        else:
            cats = sorted(df[c].dropna().unique().tolist())
            fit_categories[c] = cats
        # ``pd.Categorical`` with a fixed categories argument enforces the
        # column ordering and drops unseen categories silently — exactly the
        # behaviour we want when transforming a held-out fold.
        col_cat = pd.Categorical(df[c], categories=cats)
        ohe = pd.get_dummies(col_cat, prefix=c, dtype=float).reset_index(drop=True)
        encoded_pieces.append(ohe)

    out = pd.concat(encoded_pieces, axis=1)
    return out, fit_categories


def split_target_batch(
    df: pd.DataFrame,
    *,
    test_fraction: float,
    seed: int,
    group_aware: bool,
    group_col: str = "Composite_Mix_ID",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the target-batch DataFrame into (train, test).

    The faithful replication uses a row-level shuffle (``group_aware=False``)
    matching the paper's reported 80/20 split.  The leakage-audit variant
    uses ``group_aware=True`` and groups on Composite_Mix_ID so no formulation
    appears in both halves.

    Parameters
    ----------
    df:
        Target-batch frame returned by ``prepare_tl_split``.
    test_fraction:
        Fraction of rows (or groups) to allocate to the held-out test set.
    seed:
        Reproducibility seed for the underlying shuffle.
    group_aware:
        When True, split on unique values of ``group_col`` rather than rows.
    group_col:
        Column to group on when ``group_aware`` is True.
    """
    rng = np.random.default_rng(seed)
    if not group_aware:
        n = len(df)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_test = int(round(n * test_fraction))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return (
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
        )

    if group_col not in df.columns:
        raise KeyError(
            f"group_aware split requested but {group_col!r} not in DataFrame"
        )
    groups = df[group_col].unique()
    rng.shuffle(groups)
    n_test_groups = max(1, int(round(len(groups) * test_fraction)))
    test_groups = set(groups[:n_test_groups])
    test_mask = df[group_col].isin(test_groups)
    return (
        df.loc[~test_mask].reset_index(drop=True),
        df.loc[test_mask].reset_index(drop=True),
    )
