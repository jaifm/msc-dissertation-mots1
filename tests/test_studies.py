"""Studies framework: registry resolves, YAMLs parse, defaults merge cleanly."""
from __future__ import annotations

import importlib

import pytest

from src.studies import (
    STUDIES_CONFIG_DIR,
    _STUDY_MODULES,
    list_studies,
    load_study_params,
)


def test_registry_modules_import():
    for name, module_path in _STUDY_MODULES.items():
        module = importlib.import_module(module_path)
        assert callable(getattr(module, "run", None)), f"{name}: no run()"
        assert isinstance(getattr(module, "DEFAULTS", None), dict), f"{name}: no DEFAULTS"


@pytest.mark.parametrize("name", list_studies())
def test_params_resolve(name):
    params = load_study_params(name)
    assert isinstance(params, dict) and params, f"{name}: empty params"


def test_every_study_yaml_has_registry_entry():
    for yaml_path in STUDIES_CONFIG_DIR.glob("*.yaml"):
        assert yaml_path.stem in _STUDY_MODULES, (
            f"configs/studies/{yaml_path.name} has no registry entry in "
            f"src/studies/__init__.py"
        )


def test_overrides_take_precedence():
    params = load_study_params("loo_comparison", overrides={"limit": 3})
    assert params["limit"] == 3
    # YAML keys not overridden are preserved.
    assert params["learner"] == "xgb"


def test_gpr_alias_overrides_learner_and_output():
    params = load_study_params("loo_comparison_gpr")
    assert params["learner"] == "gpr"
    assert params["out_csv"].endswith("gpr_loo_results.csv")


def test_unknown_study_raises():
    with pytest.raises(KeyError):
        load_study_params("not_a_study")
