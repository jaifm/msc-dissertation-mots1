"""Split-conformal prediction intervals for the physics-augmented student.

Protocol
--------
1. Split training recipes (Composite_Mix_ID) into proper-train (80%) and
   calibration (20%) via group-aware shuffle.
2. Build the physics-augmented long-format dataset from proper-train only.
3. Fit the XGBoost student on proper-train augmented data.
4. For each calibration recipe: predict log1p(viscosity) at its 3 measured
   shear rates.  Nonconformity score = |log1p(y_true) − ŷ_log|.
5. Conformal quantile q(α) = ceil((n+1)(1−α))/n empirical quantile of the
   calibration scores.
6. Test coverage: fraction of (recipe, shear_rate) pairs where the true
   log1p(viscosity) falls within [ŷ_log ± q(α)].

MONDRIAN VARIANT
----------------
``mondrian: shear_rate`` computes a separate conformal quantile per measured
shear rate (a Mondrian / group-conditional conformal predictor; Vovk et al.
2005, ch. 4; Boström & Johansson 2020).  Marginal split-conformal intervals
share one radius across all shears even though the error distribution at
η(1) is much wider than at η(100); Mondrian intervals adapt the radius per
shear category, which both restores per-group coverage and shrinks the
intervals where the model is accurate.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..paths import PROJECT_ROOT

DEFAULTS: Dict[str, Any] = {
    "base_experiment": "exp_035_pa_chemistry_standardised",
    "oracle": "power_law",
    # Student model: 'xgb' (original protocol), 'ridge', or 'gpr'.
    "learner": "xgb",
    "n_aug": 30,
    "seed": 42,
    "calib_fraction": 0.20,
    "alphas": [0.10, 0.05],
    # 'none' = one marginal quantile; 'shear_rate' = one quantile per
    # measured shear (Mondrian / group-conditional conformal).
    "mondrian": "none",
    "out_csv": "data/processed/xgboost_experiments/conformal_xgb_results.csv",
    "detail_csv": "data/processed/xgboost_experiments/conformal_xgb_detail.csv",
}

_SHEAR_COLS = {
    1.0:   "Viscosity_at_shear_rate_1_1/s",
    10.0:  "Viscosity_at_shear_rate_10_1/s",
    100.0: "Viscosity_at_shear_rate_100_1/s",
}


def _group_aware_split(mix_ids: List[str], calib_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.array(mix_ids)
    rng.shuffle(ids)
    n_calib = max(1, int(round(len(ids) * calib_frac)))
    return list(ids[n_calib:]), list(ids[:n_calib])


def _long_frame(wide_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Expand a wide split frame to one row per (recipe, measured shear)."""
    rows = []
    for shear in _SHEAR_COLS:
        tmp = wide_df.copy()
        tmp["shear_rate"] = shear
        tmp["log_gamma"] = np.log10(shear)
        rows.append(tmp)
    long_df = pd.concat(rows, ignore_index=True)
    for c in feat_cols:
        if c not in long_df.columns:
            long_df[c] = 0.0
    return long_df


def _true_log(long_df: pd.DataFrame) -> np.ndarray:
    vals = long_df.apply(lambda r: r[_SHEAR_COLS[r["shear_rate"]]], axis=1)
    return np.log1p(np.maximum(vals.to_numpy(float), 0.0))


def _conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = min(max(idx, 0), n - 1)
    return float(np.sort(scores)[idx])


def run(params: Dict[str, Any]) -> pd.DataFrame:
    from ..config import load_config
    from ..physics_augmented.data import build_augmented_split, target_array
    from ..physics_augmented.learners import (
        _make_preprocessor, fit_learner, fit_xgb,
    )

    seed = int(params["seed"])
    oracle = params["oracle"]
    learner = str(params.get("learner", "xgb"))
    mondrian = params.get("mondrian", "none")
    if mondrian not in {"none", "shear_rate"}:
        raise ValueError(f"mondrian must be 'none' or 'shear_rate'; got {mondrian!r}")

    cfg = load_config(params["base_experiment"])
    cfg.physics_augmented.models = [oracle]
    cfg.physics_augmented.learners = [learner]
    cfg.physics_augmented.n_aug_points = int(params["n_aug"])
    cfg.xgb.n_trials = 0

    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)

    # Group-aware proper-train / calibration split.
    mix_ids = train_df["Composite_Mix_ID"].unique().tolist()
    proper_ids, calib_ids = _group_aware_split(mix_ids, float(params["calib_fraction"]), seed)
    proper_df = train_df[train_df["Composite_Mix_ID"].isin(proper_ids)].reset_index(drop=True)
    calib_df = train_df[train_df["Composite_Mix_ID"].isin(calib_ids)].reset_index(drop=True)

    print(f"Proper-train: {len(proper_ids)} recipes, {len(proper_df)} rows")
    print(f"Calibration:  {len(calib_ids)} recipes, {len(calib_df)} rows")
    print(f"Test:         {test_df['Composite_Mix_ID'].nunique()} recipes, {len(test_df)} rows")
    print(f"Mondrian:     {mondrian}")

    pa = cfg.physics_augmented
    split = build_augmented_split(
        proper_df, test_df, cfg,
        model_key=oracle,
        n_aug_points=pa.n_aug_points,
        shear_range=tuple(pa.shear_range),
        log_uniform=bool(pa.log_uniform_gamma),
        measured_weight=pa.measured_weight,
        seed=seed,
    )

    feat_cols = split.feature_cols
    X_train = split.train_long[feat_cols]
    y_train = target_array(split.train_long)
    w_train = split.train_long["sample_weight"].to_numpy(float)
    X_test = split.test_long[feat_cols]

    if learner == "xgb":
        fit_result = fit_xgb(
            X_train, y_train, w_train, X_test, split.test_long,
            measurement_shears=list(map(float, cfg.physics.viscosity_shears)),
            target_transform=pa.target_transform,
            cfg=cfg, seed=seed,
        )
        xgb_model = fit_result.model

        # The student was fitted inside fit_xgb with its own preprocessor;
        # rebuild an identical one (same frame, same settings) for new frames.
        preprocessor = _make_preprocessor(X_train, scale_numeric=False)
        preprocessor.fit(X_train)

        def predict_log(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                xgb_model.predict(preprocessor.transform(frame[feat_cols])), float
            )
    else:
        fit_result = fit_learner(
            learner,
            X_train=X_train, y_train=y_train, weights_train=w_train,
            X_test=X_test, test_long=split.test_long,
            measurement_shears=list(map(float, cfg.physics.viscosity_shears)),
            target_transform=pa.target_transform,
            cfg=cfg, seed=seed,
            groups=split.train_long["row_id"].to_numpy() if "row_id" in split.train_long.columns else None,
        )
        predict_pa_s = fit_result.extra["predict_fn"]

        def predict_log(frame: pd.DataFrame) -> np.ndarray:
            return np.log1p(np.maximum(predict_pa_s(frame), 0.0))

    # Nonconformity scores on the calibration recipes.
    calib_long = _long_frame(calib_df, feat_cols)
    y_pred_calib = predict_log(calib_long)
    y_true_calib = _true_log(calib_long)
    calib_scores = np.abs(y_true_calib - y_pred_calib)
    calib_shears = calib_long["shear_rate"].to_numpy(float)
    print(f"\nNonconformity scores (n={len(calib_scores)}): "
          f"mean={calib_scores.mean():.3f}, 90th pct={np.percentile(calib_scores, 90):.3f}")

    # Test predictions.
    test_long = _long_frame(test_df, feat_cols)
    y_pred_test = predict_log(test_long)
    y_true_test = _true_log(test_long)
    test_shears = test_long["shear_rate"].to_numpy(float)
    abs_err_test = np.abs(y_true_test - y_pred_test)

    coverage_rows: List[dict] = []
    detail_rows: List[dict] = []
    for alpha in params["alphas"]:
        level = int(round((1 - alpha) * 100))
        if mondrian == "shear_rate":
            q_by_group = {
                s: _conformal_q(calib_scores[np.isclose(calib_shears, s)], alpha)
                for s in _SHEAR_COLS
            }
            q_test = np.array([q_by_group[s] for s in test_shears])
            q_desc = ", ".join(f"γ̇={s:g}: q={q:.3f}" for s, q in q_by_group.items())
        else:
            q_marginal = _conformal_q(calib_scores, alpha)
            q_test = np.full(len(test_shears), q_marginal)
            q_desc = f"q={q_marginal:.3f}"

        covered = abs_err_test <= q_test
        cov = covered.mean()
        mean_width = float(2 * q_test.mean())
        print(f"  α={alpha:.2f} (nominal {level}%): {q_desc}, "
              f"coverage={cov:.3f}, mean_width={mean_width:.3f}")

        # Per-shear coverage breakdown (informative for both variants).
        for s in _SHEAR_COLS:
            m = np.isclose(test_shears, s)
            coverage_rows.append({
                "alpha": alpha, "nominal_pct": level, "mondrian": mondrian,
                "learner": learner,
                "shear_rate": s,
                "q": float(q_test[m].mean()),
                "coverage": round(float(covered[m].mean()), 4),
                "mean_width_log": round(float(2 * q_test[m].mean()), 4),
                "n_test_points": int(m.sum()),
            })
        coverage_rows.append({
            "alpha": alpha, "nominal_pct": level, "mondrian": mondrian,
            "learner": learner,
            "shear_rate": "all",
            "q": float(q_test.mean()),
            "coverage": round(float(cov), 4),
            "mean_width_log": round(mean_width, 4),
            "n_test_points": len(y_true_test),
        })

        for idx in range(len(test_long)):
            detail_rows.append({
                "mix_id": test_long.loc[idx, "Composite_Mix_ID"],
                "shear_rate": float(test_shears[idx]),
                "alpha": alpha,
                "mondrian": mondrian,
                "y_true_log": round(float(y_true_test[idx]), 4),
                "y_pred_log": round(float(y_pred_test[idx]), 4),
                "q": round(float(q_test[idx]), 4),
                "covered": bool(covered[idx]),
            })

    out = PROJECT_ROOT / params["out_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(coverage_rows)
    result_df.to_csv(out, index=False)
    print(f"\nSaved → {out}")

    detail_out = PROJECT_ROOT / params["detail_csv"]
    pd.DataFrame(detail_rows).to_csv(detail_out, index=False)
    print(f"Detail → {detail_out}")
    return result_df
