"""Shear-rate hold-out study: train on two measured shears, evaluate on the third.

The question: does physics augmentation teach the model the continuous
viscosity-shear curve shape, or does it just learn separate composition →
viscosity mappings at fixed shear rates?

If the model learned the curve shape, it should predict the held-out shear
well even though it was never in the training set.  If it only learned point
mappings, it has to extrapolate/interpolate in log_gamma space without
guidance, and performance will collapse.

For each held-out shear γ̇_h ∈ ``holdout_shears`` and each variant in
``variants`` (no augmentation vs N augmented points), the study:

  1. Builds the long-format training set from the two *remaining* measured
     shears only.  The physics oracle is fitted using only those two points —
     for the power-law oracle two points exactly determine (K, n), so there
     is no information leakage from the held-out shear into the augmented
     curve.
  2. Trains the XGBoost student in log1p space (fixed params, no Optuna).
  3. Evaluates per-shear R²(log) and R²(raw) on the test split at all three
     measured shears, flagging the held-out one.

Primary metric: R²(log) at the held-out shear.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    # Experiment config providing the split paths + physics_augmented block.
    "base_experiment": "exp_035_pa_chemistry_standardised",
    # Which measured shear rates to hold out, one study pass per entry.
    # 10.0 is interpolation between the training shears; 1.0 and 100.0 are
    # extrapolation below/above the training range.
    "holdout_shears": [10.0],
    # Student model: 'xgb' (original), 'ridge', or 'gpr'.  The XGBoost
    # student has no inductive bias along log_gamma between training
    # shears, which is what the augmentation supplies; the ridge student
    # is linear in log_gamma by construction, so comparing the two
    # separates "augmentation fixes the tree's interpolation failure"
    # from "augmentation adds information".
    "learner": "xgb",
    # Physics oracle used to generate augmented points from the two
    # remaining measured shears.
    "oracle": "power_law",
    # (name, n_aug) variants compared per held-out shear.
    "variants": [
        {"name": "no_aug", "n_aug": 0},
        {"name": "aug_30", "n_aug": 30},
    ],
    "seed": 42,
    "out_csv": "data/processed/xgboost_experiments/shear_holdout_results.csv",
}

_SHEAR_TO_COL = {
    1.0:   "Viscosity_at_shear_rate_1_1/s",
    10.0:  "Viscosity_at_shear_rate_10_1/s",
    100.0: "Viscosity_at_shear_rate_100_1/s",
}


def _r2_log(y_true, y_pred) -> float:
    from sklearn.metrics import r2_score
    yt = np.log1p(np.maximum(np.asarray(y_true, float), 0.0))
    yp = np.log1p(np.maximum(np.asarray(y_pred, float), 0.0))
    return float(r2_score(yt, yp))


def _r2_raw(y_true, y_pred) -> float:
    from sklearn.metrics import r2_score
    return float(r2_score(np.asarray(y_true, float), np.asarray(y_pred, float)))


def _run_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
    *,
    holdout_shear: float,
    oracle: str,
    n_aug: int,
    name: str,
    seed: int,
    learner: str = "xgb",
) -> List[dict]:
    from ..physics_augmented.augment import augment_train_rows, make_test_rows
    from ..physics_augmented.learners import _make_preprocessor, fit_learner

    pa = cfg.physics_augmented
    composition_cols = [
        c for c in cfg.data.composition_cols
        if c in train_df.columns and c in test_df.columns
    ]

    eval_shears = sorted(_SHEAR_TO_COL)
    train_shears = [s for s in eval_shears if s != holdout_shear]
    train_vis_cols = [_SHEAR_TO_COL[s] for s in train_shears]
    eval_vis_cols = [_SHEAR_TO_COL[s] for s in eval_shears]

    print(f"\n--- Held-out γ̇={holdout_shear:g} | variant: {name} (n_aug={n_aug}) ---")

    # Training long-format: the two remaining measured shears only.
    train_long, _stats = augment_train_rows(
        train_df,
        model_key=oracle,
        viscosity_columns=train_vis_cols,
        measurement_shears=train_shears,
        composition_cols=composition_cols,
        n_aug_points=n_aug,
        shear_range=tuple(pa.shear_range),
        log_uniform=bool(pa.log_uniform_gamma),
        seed=seed,
    )
    train_long["log_gamma"] = np.log10(train_long["shear_rate"].astype(float))
    train_long["sample_weight"] = 1.0

    print(f"  Train rows: {len(train_long)} "
          f"(measured: {(train_long['source'] == 'measured').sum()}, "
          f"augmented: {(train_long['source'] == 'augmented').sum()})")

    # Test long-format: all three measured shears.
    test_long = make_test_rows(
        test_df,
        viscosity_columns=eval_vis_cols,
        measurement_shears=eval_shears,
        composition_cols=composition_cols,
    )
    test_long["log_gamma"] = np.log10(test_long["shear_rate"].astype(float))

    feat_cols = composition_cols + ["log_gamma"]
    X_train = train_long[feat_cols]
    w_train = train_long["sample_weight"].to_numpy(float)

    if learner == "xgb":
        # Inline fixed-parameter XGBoost (the study's original protocol,
        # kept verbatim so historical results stay bit-reproducible).
        y_train = np.log1p(np.maximum(train_long["viscosity"].to_numpy(float), 0.0))
        preprocessor = _make_preprocessor(X_train, scale_numeric=False)
        X_train_enc = preprocessor.fit_transform(X_train)

        import xgboost as xgb
        params = dict(
            max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.01,
            n_estimators=400, verbosity=0, random_state=seed,
        )
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_enc, y_train, sample_weight=w_train, verbose=False)
        y_pred_all = np.expm1(model.predict(
            preprocessor.transform(test_long[feat_cols])
        ))
    else:
        result = fit_learner(
            learner,
            X_train=X_train,
            y_train=train_long["viscosity"].to_numpy(float),
            weights_train=w_train,
            X_test=test_long[feat_cols],
            test_long=test_long,
            measurement_shears=eval_shears,
            target_transform="log1p_nonnegative",
            cfg=cfg,
            seed=seed,
        )
        y_pred_all = np.asarray(result.y_pred_test, float)

    rows = []
    for shear in eval_shears:
        mask = test_long["shear_rate"] == shear
        sub = test_long[mask]
        y_pred = y_pred_all[mask.to_numpy()]
        y_true = sub["viscosity"].to_numpy(float)

        r2l = _r2_log(y_true, y_pred)
        r2r = _r2_raw(y_true, y_pred)
        held_out = shear == holdout_shear
        print(f"  γ̇={shear:6.1f}  R²(log)={r2l:.3f}  R²(raw)={r2r:.3f}"
              f"{'  ← HELD-OUT' if held_out else ''}")
        rows.append({
            "holdout_shear": holdout_shear,
            "learner": learner,
            "variant": name, "n_aug": n_aug,
            "shear_rate": shear, "held_out": held_out,
            "r2_log": round(r2l, 4), "r2_raw": round(r2r, 4),
            "n_test": int(mask.sum()),
        })
    return rows


def run(params: Dict[str, Any]) -> pd.DataFrame:
    from ..config import load_config
    from ..data.splits import load_splits

    cfg = load_config(params["base_experiment"])
    train_df, test_df = load_splits(cfg)
    print(f"Split: {len(train_df)} train rows, {len(test_df)} test rows")
    print(f"Held-out shears: {params['holdout_shears']}")

    all_rows: List[dict] = []
    for holdout_shear in params["holdout_shears"]:
        if float(holdout_shear) not in _SHEAR_TO_COL:
            raise ValueError(
                f"holdout_shear must be one of {sorted(_SHEAR_TO_COL)}; "
                f"got {holdout_shear}"
            )
        for v in params["variants"]:
            all_rows.extend(_run_variant(
                train_df, test_df, cfg,
                holdout_shear=float(holdout_shear),
                oracle=params["oracle"],
                n_aug=int(v["n_aug"]),
                name=v["name"],
                seed=int(params["seed"]),
                learner=str(params.get("learner", "xgb")),
            ))

    results = pd.DataFrame(all_rows)

    print("\n── Summary: R²(log) at each held-out shear ──")
    held = results[results["held_out"]]
    for _, row in held.iterrows():
        print(f"  γ̇={row['holdout_shear']:6.1f}  {row['variant']:10s}  "
              f"R²(log) = {row['r2_log']:.3f}")

    out = PROJECT_ROOT / params["out_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nSaved → {out}")
    return results
