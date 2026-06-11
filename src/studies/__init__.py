"""Config-driven analysis studies.

WHAT A STUDY IS
---------------
An *experiment* (``configs/experiments/*.yaml``) is a single train/evaluate
cycle on one split, dispatched through ``run_experiment``.  A *study* is an
evaluation campaign that orchestrates many experiment runs or post-processes
their outputs — e.g. the 67-fold leave-one-chemistry-out comparison, the
few-shot recovery curve, or the held-out-shear-rate test.

Studies were originally one-off scripts at the repository root.  They are
now first-class, config-driven modules so that every number in the
dissertation can be regenerated from a YAML file:

  * each study module owns a ``DEFAULTS`` dict documenting every parameter;
  * ``configs/studies/<name>.yaml`` (optional) overrides those defaults via
    the same deep-merge semantics as experiment configs;
  * outputs are written to the same summary CSVs the dissertation tables
    are built from.

USAGE
-----
::

    python -m src.cli study --name shear_holdout
    python -m src.cli list-studies

or programmatically::

    from src.studies import run_study
    run_study("loo_comparison", overrides={"limit": 2})   # smoke test
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..config import _deep_merge
from ..paths import CONFIGS_DIR

# Registry: study name → module path (lazy import keeps heavy deps optional).
_STUDY_MODULES: Dict[str, str] = {
    "shear_holdout":      "src.studies.shear_holdout",
    # Same module with the ridge student; separate YAML + output CSV.
    "shear_holdout_ridge": "src.studies.shear_holdout",
    "loo_comparison":     "src.studies.loo_comparison",
    # Same module as loo_comparison with a GPR / ridge student; their YAMLs
    # override the learner and output CSV so each pass stays independently
    # pinned.
    "loo_comparison_gpr": "src.studies.loo_comparison",
    "loo_comparison_ridge": "src.studies.loo_comparison",
    "fewshot_comparison": "src.studies.fewshot_comparison",
    "batch_comparison":   "src.studies.batch_comparison",
    "conformal":          "src.studies.conformal",
    "gpr_uq":             "src.studies.gpr_uq",
    "oracle_curve_eval":  "src.studies.oracle_curve_eval",
    "aug_sweep":          "src.studies.aug_sweep",
    "tl_chemistry":       "src.studies.tl_chemistry",
}

STUDIES_CONFIG_DIR = CONFIGS_DIR / "studies"


def list_studies() -> List[str]:
    """Return the names of all registered studies."""
    return sorted(_STUDY_MODULES)


def load_study_params(name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve the parameter dict for a study.

    Precedence (lowest → highest): module ``DEFAULTS`` → YAML file at
    ``configs/studies/<name>.yaml`` (if present) → ``overrides``.
    """
    if name not in _STUDY_MODULES:
        raise KeyError(
            f"Unknown study {name!r}. Available: {list_studies()}"
        )
    module = importlib.import_module(_STUDY_MODULES[name])
    params = copy.deepcopy(getattr(module, "DEFAULTS", {}))

    yaml_path = STUDIES_CONFIG_DIR / f"{name}.yaml"
    if yaml_path.exists():
        with yaml_path.open() as f:
            file_params = yaml.safe_load(f) or {}
        params = _deep_merge(params, file_params)

    if overrides:
        params = _deep_merge(params, overrides)
    return params


def run_study(name: str, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Run a registered study with resolved parameters and return its result."""
    params = load_study_params(name, overrides)
    module = importlib.import_module(_STUDY_MODULES[name])
    return module.run(params)
