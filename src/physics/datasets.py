"""Build per-row physics-parameter datasets.

OVERVIEW
--------
Raw experimental data contains viscosity measurements at three shear rates
per row.  Before training any machine-learning model, this module converts
each row's viscosity triple into a set of rheology model parameters by fitting
the physics models defined in ``physics/models.py``.

The result is a new DataFrame in which the three raw viscosity columns are
accompanied by new columns for the fitted parameters (e.g. eta_0, n, lambda_time
for the Carreau-Yasuda model).  Those fitted parameters become the regression
targets: the XGBoost models learn to predict them from composition features.

WHY FIT PHYSICS MODELS FIRST?
------------------------------
Predicting raw viscosity at a single shear rate is possible but limited — the
model would only capture behaviour at that one operating point.  Predicting
physics-model parameters instead gives several advantages:

  1. **Physical interpretability**: parameters like zero-shear viscosity (η_0)
     and relaxation time (λ) have clear physical meanings that are useful for
     formulation design decisions.

  2. **Extrapolation**: once η_0 and λ are predicted, the full flow curve can
     be reconstructed at any shear rate, not just the three measured ones.

  3. **Model comparison**: running multiple physics models (CY5, Power Law,
     Sisko, etc.) and comparing XGBoost prediction accuracy reveals which
     model description best captures the relationship between composition and
     rheological behaviour.

DATASET STRUCTURE
-----------------
``build_physics_dataset`` appends fitted parameter columns to a copy of the
input DataFrame and stores the target column names in
``physics_df.attrs['target_cols']``.  The ``attrs`` dictionary is pandas'
metadata mechanism: it survives column selection and copy operations but is
lost on aggregation.  Downstream code should always access target column names
through ``attrs`` rather than hard-coding them.

``build_all_physics_datasets`` fits every model listed in ``cfg.physics.models``
and returns a nested dict:
  {model_key: {'train': DataFrame, 'test': DataFrame, 'display_name': str}}

This structure lets the sweep and LOO sections iterate over models with a
single ``for model_key in cfg.physics.models`` loop.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from .models import FITTERS


def build_physics_dataset(df: pd.DataFrame, model_key: str, cfg: Config) -> pd.DataFrame:
    """Fit a physics model to every row and return a copy of ``df`` with
    the fitted parameter columns appended.

    The fitting is row-wise: each row's viscosity measurements are treated as
    an independent optimisation problem.  Rows where the fit fails fall back
    to the model's initial parameter vector (see ``models.py``), so the
    output DataFrame always has the same shape as the input.

    Parameters
    ----------
    df:
        Raw (or preprocessed) slurry DataFrame containing viscosity columns
        and composition columns.  Not modified in place.
    model_key:
        Key identifying which physics model to fit (must exist in ``FITTERS``
        and in ``cfg.physics.targets``).
    cfg:
        Experiment configuration supplying viscosity column names, shear rates,
        and target parameter names.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with one additional column per fitted parameter.
        ``physics_df.attrs['target_cols']`` lists the new column names.
        ``physics_df.attrs['model_key']`` records which model was used.

    Raises
    ------
    KeyError
        If ``model_key`` is not in the ``FITTERS`` registry.
    """
    if model_key not in FITTERS:
        raise KeyError(f"Unknown model_key {model_key!r}. Options: {sorted(FITTERS)}")

    fitter = FITTERS[model_key]
    target_cols = cfg.targets_for(model_key)
    shear_rates = np.asarray(cfg.physics.viscosity_shears)
    visc_cols = cfg.data.viscosity_columns

    # Build each output row as a dict, then construct the DataFrame once.
    # This is faster than repeated DataFrame.loc assignment for large inputs.
    rows = []
    for _, row in df.iterrows():
        visc = np.asarray([row[c] for c in visc_cols])
        params = fitter(visc, shear_rates)
        # Start with all original columns, then overwrite/add the target columns.
        new = {c: row[c] for c in df.columns}
        for i, name in enumerate(target_cols):
            new[name] = params[i]
        rows.append(new)

    physics_df = pd.DataFrame(rows)

    # Store metadata in attrs so callers don't need to hard-code column names.
    physics_df.attrs["target_cols"] = target_cols
    physics_df.attrs["model_key"] = model_key
    return physics_df


def build_all_physics_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Fit every configured physics model on both the train and test splits.

    This is a convenience wrapper that calls ``build_physics_dataset`` for
    each model in ``cfg.physics.models`` and packages the results into the
    nested dict structure consumed by the sweep and LOO experiment sections.

    The train and test splits are processed independently so that no
    information from the test set influences the fit of training-set rows.
    (Physics fitting is per-row, so there is no shared state, but keeping
    them separate enforces the principle of strict train/test isolation.)

    Parameters
    ----------
    train_df, test_df:
        Split DataFrames from ``load_splits``.
    cfg:
        Experiment config supplying the list of models to run.

    Returns
    -------
    dict
        ``{model_key: {'train': DataFrame, 'test': DataFrame, 'display_name': str}}``
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for key in cfg.physics.models:
        out[key] = {
            "train": build_physics_dataset(train_df, key, cfg),
            "test": build_physics_dataset(test_df, key, cfg),
            # Display name is stored here so plotting/reporting code can label
            # charts without needing to re-query the config.
            "display_name": cfg.display_name(key),
        }
    return out
