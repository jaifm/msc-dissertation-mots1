"""Physics-prior knobs: monotone shear constraint and crowding feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import add_crowding_feature


def test_monotone_shear_knob_parses():
    cfg = load_config("exp_040_pa_monotone_shear")
    assert cfg.physics_augmented.monotone_shear is True
    # Older configs that never mention the knob default to False.
    base = load_config("exp_035_pa_chemistry_standardised")
    assert base.physics_augmented.monotone_shear is False


def test_crowding_knob_parses():
    cfg = load_config("exp_041_pa_crowding_feature")
    assert cfg.features.crowding_phi_max == 82.0
    base = load_config("exp_035_pa_chemistry_standardised")
    assert base.features.crowding_phi_max is None


def test_crowding_feature_math():
    df = pd.DataFrame({"Solid_Content_pct": [69.0, 77.0]})
    out = add_crowding_feature(df, phi_max=82.0)
    expected = (1.0 - np.array([69.0, 77.0]) / 82.0) ** -2
    assert np.allclose(out["crowding_mp"].to_numpy(), expected)
    # The divergence shape: higher loading → strictly larger crowding term.
    assert out["crowding_mp"].iloc[1] > out["crowding_mp"].iloc[0]


def test_crowding_feature_clips_out_of_range():
    df = pd.DataFrame({"Solid_Content_pct": [81.9, 200.0]})
    out = add_crowding_feature(df, phi_max=82.0)
    assert np.isfinite(out["crowding_mp"]).all()
    # Clipped at ratio 0.97 → bounded by (1-0.97)^-2.
    assert (out["crowding_mp"] <= (1 - 0.97) ** -2 + 1e-9).all()


def test_crowding_feature_missing_column_is_noop():
    df = pd.DataFrame({"NMC_pct": [90.0]})
    out = add_crowding_feature(df, phi_max=82.0)
    assert "crowding_mp" not in out.columns


def test_monotone_constraint_vector():
    from src.physics_augmented.learners import (
        _make_preprocessor, _monotone_constraints,
    )
    X = pd.DataFrame({
        "Dispersent_Type": ["Hypermer", "Triton", "Tego"],
        "Solid_Content_pct": [69.0, 73.0, 77.0],
        "log_gamma": [0.0, 1.0, 2.0],
    })
    pre = _make_preprocessor(X, scale_numeric=False)
    pre.fit(X)
    constraints = _monotone_constraints(pre)
    names = list(pre.get_feature_names_out())
    assert len(constraints) == len(names)
    assert constraints.count(-1) == 1
    assert names[constraints.index(-1)].endswith("log_gamma")


def test_monotone_predictions_are_nonincreasing_in_shear():
    """End-to-end: a constrained student never predicts shear-thickening."""
    import xgboost as xgb
    from src.physics_augmented.learners import (
        _make_preprocessor, _monotone_constraints,
    )
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "Solid_Content_pct": rng.uniform(69, 77, n),
        "log_gamma": rng.uniform(-1, 3, n),
    })
    # Noisy shear-thinning surface.
    y = 5.0 + 0.1 * X["Solid_Content_pct"] - 1.2 * X["log_gamma"] + rng.normal(0, 0.3, n)
    pre = _make_preprocessor(X, scale_numeric=False)
    X_enc = pre.fit_transform(X)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=4, random_state=0, verbosity=0,
        monotone_constraints=_monotone_constraints(pre),
    )
    model.fit(X_enc, y)
    grid = pd.DataFrame({
        "Solid_Content_pct": np.full(50, 73.0),
        "log_gamma": np.linspace(-1, 3, 50),
    })
    preds = model.predict(pre.transform(grid))
    assert (np.diff(preds) <= 1e-9).all()
