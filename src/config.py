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
    exclude_cols: List[str] = field(default_factory=list)
    poly_degree: int = 0
    # Maximum-packing parameter for the Maron–Pierce crowding feature
    # (1 − Solid_Content_pct/φ_max)⁻², added when set (None = disabled).
    # Concentrated-suspension rheology diverges as solids loading approaches
    # maximum packing (Krieger & Dougherty 1959; Maron & Pierce 1956); the
    # raw wt% column cannot express that divergence, so it is pre-computed.
    # Must exceed the highest observed Solid_Content_pct (~77 wt%).
    crowding_phi_max: Optional[float] = None


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
    per_target_transforms: Dict[str, str] = field(default_factory=dict)


@dataclass
class LooVariant:
    id: str
    leave_out_unit: str


@dataclass
class LooCfg:
    variants: List[LooVariant]
    target_scope: str  # 'first' or 'all'
    model: str = "xgb"  # 'xgb' or 'gpr'


@dataclass
class RunCfg:
    sections: List[str]


@dataclass
class TlConformalCfg:
    """Split-conformal interval settings for the TL section.

    Attributes
    ----------
    alphas:
        Miscoverage levels.  ``[0.10, 0.05]`` reproduces the paper's
        90% and 95% intervals.
    calib_fraction:
        Fraction of each fold's training portion used as the calibration
        set when fitting the conformal quantile.  The paper does not
        specify; 0.5 is the standard literature default.
    """
    alphas: List[float] = field(default_factory=lambda: [0.10, 0.05])
    calib_fraction: float = 0.5


@dataclass
class TlCfg:
    """Transfer-learning replication configuration.

    Only consumed when ``'tl_baseline'`` appears in ``run.sections``.

    Attributes
    ----------
    models:
        Learner names to evaluate.  Subset of {'lightgbm', 'ols', 'ridge'}.
    source_batch, target_batch:
        Source_Batch values that select the pretrain and fine-tune subsets.
        Defaults match El-Dalahmeh et al. (Batch_1 → Batch_2).
    test_fraction:
        Outer 80/20 split fraction inside the target batch.
    k_folds:
        Number of inner cross-validation folds on the target_train rows.
    seed:
        Reproducibility seed for splits, OOF folds, and learner randomness.
        Falls back to ``cfg.seed`` when omitted.
    variants:
        List of variant dicts driving the matrix of TL/no-TL × chained/direct
        × in-sample/OOF × row/group-aware splits.  See ``transfer_learning.section``
        for the supported keys.
    lightgbm:
        LightGBM hyperparameters forwarded to ``LightGBMConfig`` (paper
        defaults: 500 base trees, 100 fine-tune trees, lr 0.10/0.05).
    conformal:
        Split-conformal interval settings.  See ``TlConformalCfg``.
    """
    models: List[str] = field(default_factory=lambda: ["lightgbm", "ols", "ridge"])
    source_batch: str = "Batch_1"
    target_batch: str = "Batch_2"
    test_fraction: float = 0.20
    k_folds: int = 5
    # ``kfold_scope`` controls which rows the inner cross-validation splits.
    #   "full_target"     -- K-fold across all rows of the target batch (paper
    #                        Section 3.1: "K-fold cross-validation on the
    #                        target batch").  The outer 80/20 split is ignored
    #                        and ``test_fraction`` is unused.
    #   "train_partition" -- outer 80/20 then K-fold inside the 80% train
    #                        portion (the Phase-1 default; preserved for
    #                        backwards compatibility with exp_027).
    kfold_scope: str = "full_target"
    # ``target_transform`` controls a per-fold viscosity-target preprocessing
    # step.  log1p_nonnegative maps strictly-positive targets to a real-valued
    # log-space so squared-error loss is not dominated by η(1) outliers, then
    # expm1 is applied to predictions for reporting.  The paper's protocol does
    # not specify a transform, but achieving their reported R² at η(1) is
    # implausible without one — exposing this knob lets the dissertation
    # quantify how much of their reported lift the choice explains.
    target_transform: str = "none"
    seed: Optional[int] = None
    variants: List[Dict[str, Any]] = field(default_factory=list)
    lightgbm: Dict[str, Any] = field(default_factory=dict)
    conformal: TlConformalCfg = field(default_factory=TlConformalCfg)


@dataclass
class PhysicsAugmentedCfg:
    """Physics-augmented training configuration.

    Only consumed when ``'physics_augmented'`` appears in ``run.sections``.
    Uses a chosen rheological model to interpolate the training data:
    each train row is scipy-fitted, then evaluated at N additional shear
    rates sampled log-uniformly in ``shear_range``.  The expanded
    ``(composition, log γ̇) → η`` set is fed to one or more learners.

    Attributes
    ----------
    models:
        Physics oracles to evaluate.  Any subset of
        {'power_law', 'constrained_cy', 'cy5', 'sisko', 'williamson'}.
    learners:
        ML students.  Subset of {'xgb', 'gpr'}.
    n_aug_points:
        Augmentation samples per train row (paper: 10-100; default 30).
    shear_range:
        Tuple ``(γ̇_min, γ̇_max)`` for the sampling grid.  Industrial
        characterisation typically spans ~0.1-1000 s⁻¹.
    log_uniform_gamma:
        Sample γ̇ uniformly in log10 space (recommended) vs linear.
    measured_weight:
        Sample weight applied to the original three measured points; the
        augmented points always have weight 1.  ``measured_weight=1.0``
        treats them equally; larger values privilege real measurements.
    target_transform:
        Optional log1p transform on the viscosity target.  Same options
        as elsewhere ('none' or 'log1p_nonnegative').
    monotone_shear:
        When True, the XGBoost student is constrained to be monotonically
        non-increasing in ``log_gamma``.  Battery slurries are shear-
        thinning across the industrial 0.1–1000 s⁻¹ range, so predicted
        viscosity rising with shear rate is always a model artefact; the
        constraint encodes that domain knowledge directly (theory-guided
        ML in the sense of Karpatne et al. 2017).  Ignored by the GPR
        student, which has no native monotonicity support.
    seed:
        Reproducibility seed for the augmentation sampler.  Falls back
        to ``cfg.seed`` when omitted.
    """
    models: List[str] = field(default_factory=lambda: ["power_law", "constrained_cy"])
    learners: List[str] = field(default_factory=lambda: ["xgb", "gpr"])
    n_aug_points: int = 30
    shear_range: List[float] = field(default_factory=lambda: [0.1, 1000.0])
    log_uniform_gamma: bool = True
    measured_weight: float = 1.0
    target_transform: str = "none"
    monotone_shear: bool = False
    seed: Optional[int] = None


@dataclass
class GprCfg:
    """Gaussian Process Regression configuration.

    Only consumed when 'gpr_baseline' appears in run.sections.

    Attributes
    ----------
    kernel:
        Covariance kernel: 'matern25' (default) or 'rbf'.
    n_restarts:
        Restarts for the log marginal likelihood optimiser.
    target_transform:
        Global target transform (same options as XgbCfg).
    normalize_y:
        Centre and scale the target inside GPR before kernel computations.
        Recommended True for targets that span different orders of magnitude.
    ard:
        When True, use Automatic Relevance Determination: the kernel gets one
        independent length scale per input feature rather than a single shared
        value.  ARD lets the marginal likelihood optimiser shrink length scales
        for uninformative features toward their upper bound, effectively
        down-weighting them.  Increases optimisation cost proportionally to
        the number of features but is the principled way to do feature
        selection inside a GPR.
    per_target_transforms:
        Target-name → transform override, same semantics as
        ``XgbCfg.per_target_transforms``.  Takes priority over
        ``target_transform`` for the named targets.
    """
    kernel: str
    n_restarts: int
    target_transform: str
    normalize_y: bool
    ard: bool = False
    per_target_transforms: Dict[str, str] = field(default_factory=dict)


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
    gpr: GprCfg
    tl: TlCfg = field(default_factory=TlCfg)
    physics_augmented: PhysicsAugmentedCfg = field(default_factory=PhysicsAugmentedCfg)
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


_VALID_SECTIONS = {"baseline", "sweep", "loo", "gpr_baseline", "gpr_xgb_ensemble",
                   "tl_baseline", "physics_augmented"}
_VALID_PA_PHYSICS = {"power_law", "constrained_cy", "cy5", "sisko", "williamson"}
_VALID_PA_LEARNERS = {"xgb", "gpr", "ridge"}
_VALID_TL_LEARNERS = {"lightgbm", "ols", "ridge"}
_VALID_TL_KFOLD_SCOPES = {"full_target", "train_partition"}
_VALID_TL_TARGET_TRANSFORMS = {"none", "log1p_nonnegative"}
_VALID_GPR_KERNELS = {"rbf", "matern25", "matern15", "matern25_linear"}
_VALID_TARGET_TRANSFORMS = {"none", "log1p_nonnegative", "logit"}
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
        model=loo_raw.get("model", "xgb"),
    )
    gpr_raw = raw.get("gpr") or {
        "kernel": "matern25",
        "n_restarts": 5,
        "target_transform": "none",
        "normalize_y": True,
    }
    tl_raw = raw.get("tl") or {}
    tl_conformal_raw = tl_raw.get("conformal") or {}
    pa_raw = raw.get("physics_augmented") or {}
    pa = PhysicsAugmentedCfg(
        models=list(pa_raw.get("models", ["power_law", "constrained_cy"])),
        learners=list(pa_raw.get("learners", ["xgb", "gpr"])),
        n_aug_points=int(pa_raw.get("n_aug_points", 30)),
        shear_range=list(pa_raw.get("shear_range", [0.1, 1000.0])),
        log_uniform_gamma=bool(pa_raw.get("log_uniform_gamma", True)),
        measured_weight=float(pa_raw.get("measured_weight", 1.0)),
        target_transform=str(pa_raw.get("target_transform", "none")),
        monotone_shear=bool(pa_raw.get("monotone_shear", False)),
        seed=pa_raw.get("seed"),
    )
    tl = TlCfg(
        models=list(tl_raw.get("models", ["lightgbm", "ols", "ridge"])),
        source_batch=tl_raw.get("source_batch", "Batch_1"),
        target_batch=tl_raw.get("target_batch", "Batch_2"),
        test_fraction=float(tl_raw.get("test_fraction", 0.20)),
        k_folds=int(tl_raw.get("k_folds", 5)),
        kfold_scope=tl_raw.get("kfold_scope", "full_target"),
        target_transform=tl_raw.get("target_transform", "none"),
        seed=tl_raw.get("seed"),
        variants=list(tl_raw.get("variants", [])),
        lightgbm=dict(tl_raw.get("lightgbm", {}) or {}),
        conformal=TlConformalCfg(
            alphas=list(tl_conformal_raw.get("alphas", [0.10, 0.05])),
            calib_fraction=float(tl_conformal_raw.get("calib_fraction", 0.5)),
        ),
    )
    features_raw = dict(raw["features"])
    if "crowding_phi_max" in features_raw and features_raw["crowding_phi_max"] is not None:
        features_raw["crowding_phi_max"] = float(features_raw["crowding_phi_max"])
    return Config(
        id=raw["id"],
        description=raw.get("description", ""),
        hypothesis=raw.get("hypothesis", ""),
        seed=raw["seed"],
        paths=PathsCfg(**raw["paths"]),
        data=DataCfg(**raw["data"]),
        features=FeaturesCfg(**features_raw),
        physics=PhysicsCfg(**raw["physics"]),
        xgb=XgbCfg(**raw["xgb"]),
        loo=loo,
        run=RunCfg(**raw["run"]),
        gpr=GprCfg(**gpr_raw),
        tl=tl,
        physics_augmented=pa,
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
    for tgt, mode in cfg.xgb.per_target_transforms.items():
        if mode not in _VALID_TARGET_TRANSFORMS:
            raise ValueError(
                f"Unknown per_target_transforms[{tgt!r}]={mode!r}. "
                f"Valid: {sorted(_VALID_TARGET_TRANSFORMS)}"
            )
    for model_key in cfg.physics.models:
        if model_key not in cfg.physics.targets:
            raise ValueError(f"physics.targets missing entry for model {model_key!r}")
    if cfg.loo.target_scope not in {"first", "all"}:
        raise ValueError(
            f"loo.target_scope must be 'first' or 'all' (got {cfg.loo.target_scope!r})"
        )
    if cfg.loo.model not in {"xgb", "gpr"}:
        raise ValueError(
            f"loo.model must be 'xgb' or 'gpr' (got {cfg.loo.model!r})"
        )
    if "physics_augmented" in cfg.run.sections:
        bad = set(cfg.physics_augmented.models) - _VALID_PA_PHYSICS
        if bad:
            raise ValueError(
                f"Unknown physics_augmented.models: {sorted(bad)}. "
                f"Valid: {sorted(_VALID_PA_PHYSICS)}"
            )
        bad_learn = set(cfg.physics_augmented.learners) - _VALID_PA_LEARNERS
        if bad_learn:
            raise ValueError(
                f"Unknown physics_augmented.learners: {sorted(bad_learn)}. "
                f"Valid: {sorted(_VALID_PA_LEARNERS)}"
            )
        if cfg.physics_augmented.n_aug_points < 0:
            raise ValueError(
                f"physics_augmented.n_aug_points must be >= 0; "
                f"got {cfg.physics_augmented.n_aug_points}"
            )
        sr = cfg.physics_augmented.shear_range
        if len(sr) != 2 or sr[0] <= 0 or sr[1] <= sr[0]:
            raise ValueError(
                f"physics_augmented.shear_range must be (lo, hi) with "
                f"0 < lo < hi; got {sr}"
            )
        if cfg.physics_augmented.target_transform not in {"none", "log1p_nonnegative"}:
            raise ValueError(
                f"physics_augmented.target_transform must be 'none' or "
                f"'log1p_nonnegative'; got "
                f"{cfg.physics_augmented.target_transform!r}"
            )
    if cfg.features.crowding_phi_max is not None and cfg.features.crowding_phi_max <= 0:
        raise ValueError(
            f"features.crowding_phi_max must be positive when set; "
            f"got {cfg.features.crowding_phi_max}"
        )
    if "tl_baseline" in cfg.run.sections:
        bad = set(cfg.tl.models) - _VALID_TL_LEARNERS
        if bad:
            raise ValueError(
                f"Unknown tl.models entries: {sorted(bad)}. "
                f"Valid: {sorted(_VALID_TL_LEARNERS)}"
            )
        if cfg.tl.kfold_scope not in _VALID_TL_KFOLD_SCOPES:
            raise ValueError(
                f"tl.kfold_scope must be one of {sorted(_VALID_TL_KFOLD_SCOPES)} "
                f"(got {cfg.tl.kfold_scope!r})"
            )
        if cfg.tl.target_transform not in _VALID_TL_TARGET_TRANSFORMS:
            raise ValueError(
                f"tl.target_transform must be one of "
                f"{sorted(_VALID_TL_TARGET_TRANSFORMS)} "
                f"(got {cfg.tl.target_transform!r})"
            )
        if not 0.0 < cfg.tl.test_fraction < 1.0:
            raise ValueError(
                f"tl.test_fraction must be in (0, 1); got {cfg.tl.test_fraction}"
            )
        if cfg.tl.k_folds < 2:
            raise ValueError(f"tl.k_folds must be >= 2; got {cfg.tl.k_folds}")
        for alpha in cfg.tl.conformal.alphas:
            if not 0.0 < alpha < 1.0:
                raise ValueError(
                    f"tl.conformal.alphas entries must be in (0, 1); got {alpha}"
                )
        for v in cfg.tl.variants:
            if "id" not in v:
                raise ValueError(f"tl.variants entry missing 'id': {v}")
    if "gpr_baseline" in cfg.run.sections:
        if cfg.gpr.kernel not in _VALID_GPR_KERNELS:
            raise ValueError(
                f"gpr.kernel must be one of {sorted(_VALID_GPR_KERNELS)} "
                f"(got {cfg.gpr.kernel!r})"
            )
        if cfg.gpr.target_transform not in _VALID_TARGET_TRANSFORMS:
            raise ValueError(
                f"gpr.target_transform must be one of {sorted(_VALID_TARGET_TRANSFORMS)} "
                f"(got {cfg.gpr.target_transform!r})"
            )
        for tgt, mode in cfg.gpr.per_target_transforms.items():
            if mode not in _VALID_TARGET_TRANSFORMS:
                raise ValueError(
                    f"Unknown gpr.per_target_transforms[{tgt!r}]={mode!r}. "
                    f"Valid: {sorted(_VALID_TARGET_TRANSFORMS)}"
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
