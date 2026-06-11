"""All experiment YAMLs must parse, validate, and resolve paths cleanly."""
from __future__ import annotations

import pytest

from src.config import _VALID_SECTIONS, list_experiments, load_config


def test_base_loads():
    cfg = load_config("base")
    assert cfg.id == "base"
    assert cfg.seed == 42
    assert set(cfg.physics.models) >= {"cy5", "power_law"}


@pytest.mark.parametrize("name", list_experiments())
def test_every_experiment_parses(name):
    cfg = load_config(name)
    assert cfg.id == name
    # Sections must be a non-empty subset of the allowed set.
    assert set(cfg.run.sections) <= _VALID_SECTIONS
    assert cfg.run.sections, f"{name} has no sections"
    # Every listed model must have a target list.
    for model_key in cfg.physics.models:
        assert cfg.physics.targets[model_key], f"{name}: no targets for {model_key}"


def test_override_semantics():
    base = load_config("base")
    exp = load_config("exp_002_trials_20")
    assert exp.xgb.n_trials == 20
    assert exp.xgb.n_estimators == base.xgb.n_estimators  # not overridden
    assert exp.features.use_engineered == base.features.use_engineered  # dict merge
