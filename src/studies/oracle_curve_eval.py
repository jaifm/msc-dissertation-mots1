"""Oracle-as-ground-truth evaluation across a dense shear-rate grid.

IMPORTANT CAVEAT — EXPLICITLY STATED
------------------------------------
This evaluation uses the physics oracle (fitted to the three measured
viscosities per test row) as pseudo-ground-truth at intermediate and extreme
shear rates.  No rheometer measurements exist outside γ̇ ∈ {1, 10, 100} s⁻¹.
The R² figures here are NOT equivalent to measured R².  They quantify
whether the ML model has learned the SHAPE of the viscosity curve as
described by the physics model — not whether it matches true rheology at
untested shear rates.  This is appropriate for comparing models across the
full shear-rate range and for diagnosing extrapolation behaviour.  These
figures must never be reported as performance claims against real
experimental data.

WHAT THIS DOES
--------------
For the fixed chemistry split:
  1. Trains the XGBoost physics-augmented model (log1p target, n_aug points)
     on the training set.
  2. Fits the chosen oracle to each test row's 3 measurements → parameters.
  3. Evaluates ML model and oracle at ``n_eval_shears`` log-uniform shear
     rates across ``shear_range``.
  4. Reports R²(oracle-gt) per shear decade and writes the full table.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    "base_experiment": "exp_030_pa_log_target",
    "split_dir": "data/processed/splits_chemistry/default",
    "oracle_models": ["power_law", "cy5"],
    # Student: 'xgb' keeps the study's original inline XGBoost (canonical,
    # bit-reproducible); 'ridge' / 'gpr' route through the shared
    # physics-augmented learners.
    "learner": "xgb",
    # Apply the shear-thinning monotone constraint to the inline XGBoost
    # (only meaningful when learner == 'xgb').
    "monotone_shear": False,
    "n_aug": 30,
    "n_eval_shears": 200,
    "shear_range": [0.1, 1000.0],
    "out_csv": "data/processed/xgboost_experiments/oracle_curve_eval.csv",
    "bands_csv": "data/processed/xgboost_experiments/oracle_curve_bands.csv",
}

CAVEAT = (
    "\n*** CAVEAT: oracle used as pseudo-ground-truth — NOT measured data. ***\n"
    "    R² figures quantify curve-shape recovery, not true rheological accuracy.\n"
    "    Do NOT report these as performance claims against experiment.\n"
)

BANDS = [
    (0.1,    1.0,   "0.1–1   (below measured)"),
    (1.0,   10.0,   "1–10    (measured range)"),
    (10.0, 100.0,   "10–100  (measured range)"),
    (100.0, 1000.0, "100–1000 (above measured)"),
]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def _run_oracle(oracle_key: str, eval_shears: np.ndarray, params: Dict[str, Any]) -> pd.DataFrame:
    import xgboost as xgb

    from ..config import load_config
    from ..physics_augmented.augment import fit_row, forward
    from ..physics_augmented.data import (
        build_augmented_split, feature_matrix, target_array, weight_array,
    )
    from ..physics_augmented.learners import _make_preprocessor

    split_dir = PROJECT_ROOT / params["split_dir"]
    train_df = pd.read_csv(split_dir / "train.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    cfg = load_config(params["base_experiment"])
    cfg.physics_augmented.n_aug_points = int(params["n_aug"])
    cfg.physics_augmented.models = [oracle_key]

    pa = cfg.physics_augmented
    split = build_augmented_split(
        train_df, test_df, cfg,
        model_key=oracle_key,
        n_aug_points=pa.n_aug_points,
        shear_range=tuple(pa.shear_range),
        log_uniform=pa.log_uniform_gamma,
        measured_weight=pa.measured_weight,
        seed=cfg.seed,
    )

    X_train = feature_matrix(split.train_long, split.feature_cols)
    y_train = target_array(split.train_long)
    w_train = weight_array(split.train_long)
    feat_cols = split.feature_cols  # composition cols + ["log_gamma"]

    learner = str(params.get("learner", "xgb"))
    if learner == "xgb":
        # Inline fixed-parameter XGBoost (the study's original protocol,
        # kept verbatim so historical results stay bit-reproducible).
        preprocessor = _make_preprocessor(X_train, scale_numeric=False)
        X_enc = preprocessor.fit_transform(X_train)
        y_fit = np.log1p(np.maximum(y_train, 0.0))

        xgb_params = dict(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        if params.get("monotone_shear"):
            from ..physics_augmented.learners import _monotone_constraints
            xgb_params["monotone_constraints"] = _monotone_constraints(preprocessor)
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_enc, y_fit, sample_weight=w_train)

        def predict_log(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                model.predict(preprocessor.transform(frame[feat_cols])), float
            )
    else:
        from ..physics_augmented.learners import fit_learner
        result = fit_learner(
            learner,
            X_train=X_train,
            y_train=y_train,
            weights_train=w_train,
            X_test=feature_matrix(split.test_long, split.feature_cols),
            test_long=split.test_long,
            measurement_shears=[1.0, 10.0, 100.0],
            target_transform="log1p_nonnegative",
            cfg=cfg,
            seed=cfg.seed,
        )
        predict_pa_s = result.extra["predict_fn"]

        def predict_log(frame: pd.DataFrame) -> np.ndarray:
            return np.log1p(np.maximum(predict_pa_s(frame), 0.0))

    print(f"  Trained {learner} on {len(X_train)} rows "
          f"({split.stats.n_fit_success}/{split.stats.n_train_rows} fits ok)")

    comp_cols = [c for c in feat_cols if c != "log_gamma"]
    meas_shears = [1.0, 10.0, 100.0]
    visc_cols = [
        "Viscosity_at_shear_rate_1_1/s",
        "Viscosity_at_shear_rate_10_1/s",
        "Viscosity_at_shear_rate_100_1/s",
    ]

    records: List[dict] = []
    n_fit_ok = 0

    for _, recipe in test_df.iterrows():
        viscosities = [float(recipe[c]) for c in visc_cols]
        try:
            fit_params = fit_row(oracle_key, viscosities, meas_shears)
        except Exception:
            continue
        n_fit_ok += 1

        comp_vals = {c: recipe[c] for c in comp_cols if c in recipe.index}
        rows = []
        for gamma in eval_shears:
            row = dict(comp_vals)
            row["log_gamma"] = float(np.log10(gamma))
            rows.append(row)

        X_eval = pd.DataFrame(rows)[feat_cols]
        ml_log_preds = predict_log(X_eval)

        for gamma, ml_log in zip(eval_shears, ml_log_preds):
            oracle_eta = float(forward(oracle_key, fit_params, np.array([gamma]))[0])
            records.append({
                "Composite_Mix_ID": recipe["Composite_Mix_ID"],
                "oracle_model":     oracle_key,
                "shear_rate_eval":  float(gamma),
                "oracle_eta":       oracle_eta,
                "oracle_log":       float(np.log1p(max(oracle_eta, 0.0))),
                "ml_log_pred":      float(ml_log),
                "ml_eta_pred":      float(np.expm1(float(ml_log))),
            })

    print(f"  Oracle fits: {n_fit_ok}/{len(test_df)} test rows succeeded")
    return pd.DataFrame(records)


def run(params: Dict[str, Any]) -> pd.DataFrame:
    print(CAVEAT)
    print("=" * 72)
    print("Oracle-as-ground-truth curve evaluation")
    print("=" * 72)

    lo, hi = map(float, params["shear_range"])
    eval_shears = np.logspace(np.log10(lo), np.log10(hi), int(params["n_eval_shears"]))
    print(f"Evaluation grid: {len(eval_shears)} shear rates, {lo}–{hi} s⁻¹\n")

    all_dfs: List[pd.DataFrame] = []
    band_records: List[dict] = []

    for oracle_key in params["oracle_models"]:
        print(f"--- Oracle: {oracle_key} ---")
        df = _run_oracle(oracle_key, eval_shears, params)
        if df.empty:
            print("  No results.")
            continue
        all_dfs.append(df)

        print(f"\n  {'Shear band':<28} {'N':>6}  {'R²(oracle-gt)':>14}")
        print("  " + "-" * 52)
        for band_lo, band_hi, label in BANDS:
            mask = (df["shear_rate_eval"] >= band_lo) & (df["shear_rate_eval"] < band_hi)
            sub = df[mask]
            if len(sub) < 2:
                continue
            r2 = _r2(sub["oracle_log"].to_numpy(), sub["ml_log_pred"].to_numpy())
            measured = band_lo >= 1.0 and band_hi <= 100.0
            flag = " ← measured range" if measured else ""
            print(f"  {label:<28} {len(sub):>6}  {r2:>14.3f}{flag}")
            band_records.append({
                "oracle_model":      oracle_key,
                "shear_band":        label,
                "shear_lo":          band_lo,
                "shear_hi":          band_hi,
                "n":                 len(sub),
                "r2_oracle_gt":      r2,
                "is_measured_range": measured,
            })
        print()

    out_csv = PROJECT_ROOT / params["out_csv"]
    bands_csv = PROJECT_ROOT / params["bands_csv"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_dfs:
        pd.concat(all_dfs, ignore_index=True).to_csv(out_csv, index=False)
    bands_df = pd.DataFrame(band_records)
    if not bands_df.empty:
        bands_df.to_csv(bands_csv, index=False)

    print(CAVEAT)
    print(f"Full predictions → {out_csv}")
    print(f"Band summary     → {bands_csv}")
    return bands_df
