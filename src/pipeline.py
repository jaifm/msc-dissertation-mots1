"""Sklearn preprocessing pipeline and scalar metrics helpers.

OVERVIEW
--------
This module builds the feature preprocessing pipeline that sits between the
raw/engineered composition DataFrame and the XGBoost model.  It is the single
place where column typing, imputation, and optional log-transformation are
configured, so changing any of those decisions here propagates to every
training and evaluation path in the project.

DESIGN DECISIONS
----------------
1.  **ColumnTransformer with explicit cat/num split.**
    Composition data contains one categorical column (``Dispersent_Type``) and
    many numeric columns.  A single sklearn ``Pipeline`` cannot handle mixed
    types cleanly, so a ``ColumnTransformer`` is used to route each column to
    the appropriate transformer.

2.  **Dtype-based routing.**
    Column roles are inferred from pandas dtypes: numeric columns (float64,
    int64, etc., excluding bool) go to the numeric branch; everything else —
    including the string ``Dispersent_Type`` column — goes to the categorical
    branch.  This means adding a new column to the DataFrame automatically
    routes it correctly without any manual list maintenance.

3.  **Median imputation for numerics.**
    A small number of rows may have NaN for engineered features (e.g. when a
    component percentage is absent).  Median imputation is more robust than
    mean imputation for skewed small-sample distributions and does not require
    an explicit missing-value indicator column.

4.  **Optional log1p transform.**
    Some experiments pass ``log_numeric_features=True`` to compare whether
    compressing the numeric scale improves model performance.  The transform
    is added as an extra pipeline step so the imputation always runs on the
    raw scale (log of a NaN-filled value is still defined).

5.  **remainder='drop'.**
    Any column not explicitly listed in ``numeric_cols`` or ``categorical_cols``
    (e.g. target columns that accidentally appear in X) is silently dropped.
    This prevents data leakage if the caller passes a wider DataFrame.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures


def make_preprocessor(
    X_df: pd.DataFrame,
    log_numeric_features: bool = False,
    poly_degree: int = 0,
) -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot-encodes categoricals and
    imputes (+optionally log1p-transforms) numerics.

    Column roles are inferred from dtype at call time:
    - Numeric dtypes (excluding bool) → median impute, optional log1p
    - Everything else (strings, object, categorical) → one-hot encode

    Parameters
    ----------
    X_df:
        Feature DataFrame whose column dtypes determine routing.  The
        ColumnTransformer is fitted on the same DataFrame later; passing a
        representative sample here is sufficient for routing purposes.
    log_numeric_features:
        When True, appends a log1p step after imputation for all numeric
        columns.  Useful for right-skewed features (see the sweep section).
    poly_degree:
        When > 1, appends a ``PolynomialFeatures(degree=poly_degree,
        include_bias=False)`` step to the numeric sub-pipeline after
        imputation.  Applied only to numeric columns; OHE output is
        unaffected.  Degree 2 with N numeric features adds N² terms
        (squared + pairwise products), making interactions explicit for
        XGBoost single-split capture.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessor ready for ``fit_transform`` / ``transform``.
    """
    # Identify numeric columns: any dtype that sklearn/numpy treats as a
    # floating-point or integer type, excluding booleans (which are technically
    # numeric in numpy but represent flags, not magnitudes).
    numeric_cols = [
        c for c in X_df.columns if is_numeric_dtype(X_df[c]) and not is_bool_dtype(X_df[c])
    ]
    # Everything that is not numeric is treated as categorical.  In practice
    # this is just Dispersent_Type, but the logic handles any string column
    # without requiring a hard-coded exclusion list.
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    transformers = []

    if categorical_cols:
        # OneHotEncoder produces one binary column per category level.
        # handle_unknown='ignore' is required for test rows that carry a
        # dispersant type not seen in the training fold.
        transformers.append(
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            )
        )

    if numeric_cols:
        # Median imputation first so that any NaN introduced by the feature
        # engineering step (e.g. a ratio whose denominator column is absent)
        # is filled before the optional log transform sees the data.
        steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]

        if log_numeric_features:
            # log1p(x) = log(1 + x) is safe for x >= 0 and compresses
            # high-value outliers that can dominate tree split selection.
            # feature_names_out='one-to-one' preserves column names so that
            # downstream feature-importance reporting stays readable.
            steps.append(
                (
                    "log1p",
                    FunctionTransformer(
                        np.log1p, validate=False, feature_names_out="one-to-one"
                    ),
                )
            )

        if poly_degree > 1:
            # Degree-2 expansion: adds x_i^2 and x_i*x_j terms for all
            # numeric pairs.  include_bias=False omits the constant column
            # (XGBoost has its own bias via the tree intercept).
            # Applied after imputation so no NaN reaches the polynomial step.
            steps.append(
                ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False))
            )

        transformers.append(("num", Pipeline(steps), numeric_cols))

    # remainder='drop' ensures that target columns or any other column not
    # explicitly routed above cannot leak into the feature matrix.
    return ColumnTransformer(transformers=transformers, remainder="drop")


def r2_mse_rmse(y_true, y_pred) -> Tuple[float, float, float]:
    """Compute three complementary regression metrics in one call.

    Returns (R², MSE, RMSE) as plain Python floats so results can be JSON
    serialised without further conversion.

    R² is the primary ranking metric; RMSE is reported in the same units as
    the target (e.g. Pa·s for viscosity parameters) and is therefore the
    most interpretable error for domain experts.  MSE is included because it
    is the loss function optimised internally by XGBoost.
    """
    r2 = float(r2_score(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return r2, mse, rmse
