"""Experiment configuration loading, merging, and validation.

OVERVIEW
--------
Each experiment is defined by a YAML file under ``configs/experiments/``.
Rather than duplicating shared settings across every file, experiment YAMLs
inherit from ``configs/base.yaml`` via a ``defaults: base`` key and only
specify the values they want to override.  The loader deep-merges the override
dict onto the base dict before constructing the typed ``Config`` dataclass.

WHY TYPED DATACLASSES?
-----------------------
Storing configuration as a plain dict (as many projects do) means that
misspelled keys silently return ``None`` or raise a ``KeyError`` at the point
of use, which can be far from where the config was loaded.  Typed dataclasses
catch misspellings at construction time and enable IDE auto-complete, making
it harder to write experiments with invalid settings.

A third-party library like pydantic would provide additional validation
(e.g. value ranges), but introduces an extra dependency that may not be
available in a restricted lab environment.  The current approach balances
correctness against portability.

MERGE SEMANTICS
---------------
Deep-merge rules:
  - dict + dict  →  merge key-by-key recursively
  - anything else → the override value replaces the base value

Lists are NOT concatenated; the override list replaces the base list
entirely.  This is intentional: if ``base.yaml`` defines five physics models
and an experiment YAML specifies two, it should run two models, not seven.

VALIDATION
----------
``_validate`` checks a small set of invariants that would cause silent
failures downstream:
  - ``run.sections`` must be a subset of the known section names
  - ``xgb.target_transform`` must be one of the implemented modes
  - Every entry in ``physics.models`` must have a corresponding entry in
    ``physics.targets``
  - ``loo.target_scope`` must be either 'first' or 'all'
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import CONFIGS_DIR, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Typed sub-configs
# ---------------------------------------------------------------------------


@dataclass
class PathsCfg:
    raw: str               # directory containing the raw batch CSVs
    splits: str            # directory containing the pre-computed split CSVs
    train_csv: str
    test_csv: str
    experiments_root: str


@dataclass
class DataCfg:
    group_column: str
    viscosity_columns: List[str]
    composition_cols: List[str]


@dataclass
class FeaturesCfg:
    use_engineered: bool
    log_numeric: bool


@dataclass
class PhysicsCfg:
    viscosity_shears: List[float]
    models: List[str]
    display_names: Dict[str, str]
    targets: Dict[str, List[str]]


@dataclass
class XgbCfg:
    n_trials: int
    n_estimators: int
    target_transform: str
    # XGBoost objective function.  Common options:
    #   reg:squarederror      -- standard MSE (default; sensitive to outliers)
    #   reg:pseudohubererror  -- pseudo Huber loss; robust to outlier targets
    #                            (failed physics fits fall back to initial params
    #                             and become outliers; this objective down-weights them)
    # Use objective_params to pass extra arguments, e.g. {huber_slope: 1.0}.
    objective: str
    objective_params: Dict[str, Any]
    fixed_params: Dict[str, Any]


@dataclass
class LooVariant:
    id: str
    leave_out_unit: str


@dataclass
class LooCfg:
    variants: List[LooVariant]
    target_scope: str  # 'first' or 'all'


@dataclass
class RunCfg:
    sections: List[str]


@dataclass
class Config:
    id: str
    description: str
    hypothesis: str
    seed: int
    paths: PathsCfg
    data: DataCfg
    features: FeaturesCfg
    physics: PhysicsCfg
    xgb: XgbCfg
    loo: LooCfg
    run: RunCfg
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- convenience -------------------------------------------------------

    def display_name(self, model_key: str) -> str:
        """Human-friendly label for a model key."""
        return self.physics.display_names.get(model_key, model_key)

    def targets_for(self, model_key: str) -> List[str]:
        return self.physics.targets[model_key]

    def resolve(self, relative: str) -> Path:
        """Resolve a path key relative to the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.paths.raw)

    @property
    def splits_dir(self) -> Path:
        return self.resolve(self.paths.splits)

    @property
    def train_path(self) -> Path:
        return self.splits_dir / self.paths.train_csv

    @property
    def test_path(self) -> Path:
        return self.splits_dir / self.paths.test_csv

    @property
    def experiments_root_path(self) -> Path:
        return self.resolve(self.paths.experiments_root)

    @property
    def results_dir(self) -> Path:
        return self.experiments_root_path / self.id

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view for JSON serialization (results snapshot)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_VALID_SECTIONS = {"baseline", "sweep", "loo"}
_VALID_TARGET_TRANSFORMS = {"none", "log1p_nonnegative"}
_VALID_OBJECTIVES = {"reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict that merges ``override`` onto ``base`` recursively.

    Rules:
      - dict vs dict     → merge key-by-key
      - anything else    → override replaces base (lists are NOT concatenated)
    """
    out = copy.deepcopy(base)
    for key, ov in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(ov, dict):
            out[key] = _deep_merge(out[key], ov)
        else:
            out[key] = copy.deepcopy(ov)
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _build(raw: Dict[str, Any]) -> Config:
    loo_raw = raw["loo"]
    loo = LooCfg(
        variants=[LooVariant(**v) for v in loo_raw["variants"]],
        target_scope=loo_raw["target_scope"],
    )
    return Config(
        id=raw["id"],
        description=raw.get("description", ""),
        hypothesis=raw.get("hypothesis", ""),
        seed=raw["seed"],
        paths=PathsCfg(**raw["paths"]),
        data=DataCfg(**raw["data"]),
        features=FeaturesCfg(**raw["features"]),
        physics=PhysicsCfg(**raw["physics"]),
        xgb=XgbCfg(**raw["xgb"]),
        loo=loo,
        run=RunCfg(**raw["run"]),
    )


def _validate(cfg: Config) -> None:
    bad_sections = set(cfg.run.sections) - _VALID_SECTIONS
    if bad_sections:
        raise ValueError(
            f"Unknown run.sections: {sorted(bad_sections)}. "
            f"Valid options: {sorted(_VALID_SECTIONS)}"
        )
    if cfg.xgb.target_transform not in _VALID_TARGET_TRANSFORMS:
        raise ValueError(
            f"Unknown xgb.target_transform={cfg.xgb.target_transform!r}. "
            f"Valid: {sorted(_VALID_TARGET_TRANSFORMS)}"
        )
    if cfg.xgb.objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"Unknown xgb.objective={cfg.xgb.objective!r}. "
            f"Valid: {sorted(_VALID_OBJECTIVES)}"
        )
    for model_key in cfg.physics.models:
        if model_key not in cfg.physics.targets:
            raise ValueError(f"physics.targets missing entry for model {model_key!r}")
    if cfg.loo.target_scope not in {"first", "all"}:
        raise ValueError(
            f"loo.target_scope must be 'first' or 'all' (got {cfg.loo.target_scope!r})"
        )


def load_config(name: str, configs_dir: Optional[Path] = None) -> Config:
    """Load an experiment config by name.

    ``name`` is either ``'base'`` or the stem of a file under
    ``configs/experiments/`` (e.g. ``'exp_001_baseline'``). Extra ``.yaml``
    suffix is tolerated.
    """
    root = Path(configs_dir) if configs_dir else CONFIGS_DIR
    name = name.removesuffix(".yaml")

    base_raw = _read_yaml(root / "base.yaml")
    if name == "base":
        merged = base_raw
    else:
        exp_raw = _read_yaml(root / "experiments" / f"{name}.yaml")
        parent = exp_raw.pop("defaults", "base")
        if parent != "base":
            raise ValueError(
                f"Only 'defaults: base' is supported today (got {parent!r} in {name})."
            )
        merged = _deep_merge(base_raw, exp_raw)

    cfg = _build(merged)
    _validate(cfg)
    return cfg


def list_experiments(configs_dir: Optional[Path] = None) -> List[str]:
    """Return the ids of all experiment YAMLs."""
    root = Path(configs_dir) if configs_dir else CONFIGS_DIR
    exp_dir = root / "experiments"
    if not exp_dir.exists():
        return []
    return sorted(p.stem for p in exp_dir.glob("*.yaml"))
