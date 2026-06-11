# Reproducibility Guide

Every number in RESULTS.md and the dissertation can be regenerated from
this repository.  This document records how.

## Environment

```bash
conda env create -f environment.yml          # creates msc-dissertation-mots1-v2
conda activate msc-dissertation-mots1-v2
pip install lightgbm                         # TL replication only
pytest tests/ -q                             # should pass clean
```

Python 3.11; the pinned dependency families are numpy/pandas/scipy/
scikit-learn/xgboost/optuna (see `environment.yml`).

## The three layers of reproducibility

### 1. Fixed, hash-pinned splits

All headline evaluation uses the split files under
`data/processed/splits_chemistry/`, generated once by
`scripts/prepare_chemistry_splits.py` and pinned by SHA-256 in
`MANIFEST.json` (132 splits: default, 67 LOO, few-shot k∈{1,2}, two
cross-batch).  **These files are immutable** — regenerating them changes
the hashes and invalidates every downstream comparison.  Only the
prepare script may write to that directory; `--batch-only` refreshes the
two cross-batch splits without touching the rest (verified
byte-identical).

Verify integrity at any time:

```bash
python - <<'EOF'
import hashlib, json, pathlib
root = pathlib.Path("data/processed/splits_chemistry")
manifest = json.loads((root / "MANIFEST.json").read_text())
bad = 0
for e in manifest["splits"]:
    for kind in ("train", "test"):
        p = root / e["name"] / f"{kind}.csv"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != e[f"{kind}_sha256"]:
            bad += 1; print("MISMATCH:", p)
print("OK" if not bad else f"{bad} mismatches")
EOF
```

### 2. Config-first experiments

Every experiment is one YAML under `configs/experiments/`, inheriting
`configs/base.yaml` (deep merge; lists/scalars replace, dicts merge).
The loader (`src/config.py`) validates against typed dataclasses at load
time.  Run:

```bash
python -m src.cli run --config exp_044_pa_ridge_student
python -m src.cli list                       # all experiment ids
```

Each run gets a unique `run_id` (`<experiment_id>_<timestamp>`), never
overwrites, snapshots its merged config to `config_snapshot.json` in its
run directory, and appends to
`data/processed/xgboost_experiments/experiment_manifest.json`.
Run directories are local artifacts (gitignored); the manifest and the
top-level summary CSVs are tracked.

### 3. Config-first studies

Multi-run evaluation campaigns (LOO, few-shot, cross-batch, held-out
shear, conformal, UQ, curve evaluation) are *studies*: modules in
`src/studies/` with documented `DEFAULTS`, overridden by
`configs/studies/<name>.yaml`.  Run:

```bash
python -m src.cli list-studies
python -m src.cli study --name shear_holdout
```

Each study writes its summary CSV into
`data/processed/xgboost_experiments/` (tracked in git).  Registry
aliases (`loo_comparison_gpr`, `loo_comparison_ridge`,
`shear_holdout_ridge`) reuse a module with a different learner and
output file, so every published variant stays independently pinned.

## Mapping headline numbers → commands

| Number | Source CSV | Regenerate with |
|---|---|---|
| Strict hold-out R²(log) 0.802 (ridge) | run dir of exp_044 | `python -m src.cli run --config exp_044_pa_ridge_student` |
| Strict hold-out R²(log) 0.753 (XGB) | run dir of exp_035 | `python -m src.cli run --config exp_035_pa_chemistry_standardised` |
| Held-out-shear table | `shear_holdout_results.csv`, `shear_holdout_ridge_results.csv` | `python -m src.cli study --name shear_holdout` / `shear_holdout_ridge` |
| TL head-to-head | `tl_chemistry_results.csv` | `python -m src.cli study --name tl_chemistry` |
| LOO medians | `loo_comparison_results.csv`, `loo_ridge_results.csv`, `gpr_loo_results.csv` | `python -m src.cli study --name loo_comparison[...]` |
| Few-shot recovery | `fewshot_comparison_results.csv` | `python -m src.cli study --name fewshot_comparison` |
| Cross-batch | `batch_comparison_results.csv` | `python -m src.cli study --name batch_comparison` |
| Conformal coverage | `conformal_*_results.csv` | `python -m src.cli study --name conformal` (+ overrides in module docstring) |
| Curve-shape bands | `oracle_curve_bands*.csv` | `python -m src.cli study --name oracle_curve_eval` |
| GPR UQ / active learning | `gpr_uq_*.csv` | `python -m src.cli study --name gpr_uq` |

## Determinism notes

- Global seed 42 throughout (configs, split generation, study defaults).
- XGBoost single-machine training with fixed `random_state` is
  deterministic for a given library version; LightGBM runs with
  `deterministic: true, force_row_wise: true`.
- GPR results depend on scikit-learn's L-BFGS restarts (seeded); kernel
  warnings on the strict split are expected and documented in exp_037.
- The conformal and UQ studies derive calibration splits from seeded
  `numpy.random.default_rng`.

## Tests

`pytest tests/` covers: every experiment YAML parses and validates, the
deep-merge semantics, the no-leakage assertion, study registry/YAML
consistency, and the physics-prior knobs (monotone constraint vector,
crowding feature math, end-to-end monotonicity of constrained
predictions).
