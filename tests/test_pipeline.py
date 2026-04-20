from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline import make_preprocessor


def test_make_preprocessor_routes_non_numeric_to_categorical():
    # A pandas categorical column containing strings must not go through
    # median imputation in the numeric pipeline.
    X = pd.DataFrame(
        {
            "solids": [55.0, np.nan, 60.0],
            "binder_name": pd.Series(["Hypermer", "PVDF", "CMC"], dtype="category"),
        }
    )

    preprocessor = make_preprocessor(X)
    Xt = preprocessor.fit_transform(X)

    assert Xt.shape[0] == len(X)
    assert "num" in preprocessor.named_transformers_
    assert "cat" in preprocessor.named_transformers_
