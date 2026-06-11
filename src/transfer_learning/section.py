"""Section runner for the transfer-learning replication.

This is the only module in the subpackage that the main pipeline's runner
imports.  It exposes a single function, ``run_tl_baseline``, that mirrors
the signature the runner expects from other section dispatchers.

WHAT THIS RUNS
--------------
For every learner listed in ``cfg.tl.models`` and every active variant in
``cfg.tl.variants``, this function:

  1. Loads the master dataset and partitions it by Source_Batch.
  2. Splits the target batch into a training/test pair (row-level by
     default, group-aware for the leakage-audit variant).
  3. Encodes the four paper inputs with categorical one-hot encoding fitted
     on the union of source and target_train rows so test categories never
     widen the encoding.
  4. Iterates over k folds inside the target_train rows and, for each fold:
       a. Optionally splits the fold's training portion into proper-train +
          calibration for split-conformal intervals.
       b. Fits the chained η(10) → η(1)/η(100) pipeline (or the direct
          variant if requested) on (source ∪ target_train_fold).
       c. Predicts on the held-out fold.
  5. Aggregates per-fold metrics and conformal coverage statistics.

VARIANTS
--------
Each variant is a small dict in ``cfg.tl.variants`` with keys:
  * ``id``                  -- short identifier used as the variant column
  * ``transfer``            -- True/False — whether to run TL fine-tuning
  * ``chained``             -- True/False — chained vs direct strategy
  * ``use_oof``             -- True/False — paper-faithful or OOF chaining
  * ``group_aware_split``   -- True/False — group-aware vs row-level test split

Defaults (in base.yaml) configure five variants covering the matrix the
paper-replication and Phase 2 leakage audit need:

  tl_paper_chained        -- TL on, chained, in-sample (the published)
  tl_paper_chained_oof    -- TL on, chained, OOF (leakage audit a)
  tl_paper_direct         -- TL on, direct (leakage audit b)
  no_tl_chained           -- TL off, chained                — negative control
  tl_paper_chained_groups -- TL on, chained, OOF, group-aware (audit c)

All run together in one experiment by default; toggle ``active: false`` on
any variant to drop it from a particular run.

OUTPUT
------
Returns a ``SectionResult`` named ``'tl_baseline'`` with:
  * ``model_summary``   -- one row per (learner, variant) with means
  * ``per_target``      -- one row per (learner, variant, target) with means
  * ``predictions``     -- long-format predictions across folds + variants
  * ``fold_summary``    -- raw per-fold rows with conformal columns
  * ``curve_summary``   -- always None for this section (TL has no notion
                           of a viscosity curve outside the three measured
                           shears, so curve-space RMSE is not applicable)

The ``conformal_summary`` lives inside ``per_target`` as additional
``coverage_90 / mean_width_90 / coverage_95 / mean_width_95`` columns so
downstream code that already understands ``per_target`` can render them
without extra plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..config import Config
from ..experiments.results import SectionResult
from .chained import (
    ChainedFitResult,
    DirectFitResult,
    fit_chained,
    fit_direct,
    predict_chained,
    predict_direct,
)
from .conformal import (
    ConformalReport,
    conformal_coverage,
    conformal_intervals,
    split_conformal_quantile,
    split_proper_calib,
)
from .data import (
    PAPER_INPUT_COLS,
    PAPER_VISCOSITY_COLS,
    one_hot_encode_inputs,
    prepare_tl_split,
    split_target_batch,
)
from .evaluation import (
    FoldRow,
    aggregate_folds,
    build_long_predictions,
    score_predictions,
)
from .learners import _VALID_LEARNERS


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------


@dataclass
class _Variant:
    """A single ML/data configuration to evaluate in the section.

    The data-routing matrix is determined by ``(transfer, finetune)``:

      transfer=True,  finetune=True   -> Paper's TL: pretrain on Batch_1,
                                         continuation-train on Batch_2 fold.
      transfer=True,  finetune=False  -> Paper's true no-TL baseline: train
                                         only on Batch_1, predict on Batch_2
                                         without fine-tuning.  This is what
                                         Fig. 28 of El-Dalahmeh et al. calls
                                         "Train B1 → Predict B2 (no TL)".
      transfer=False, finetune=True   -> Batch-2-only training (no pretrain).
      transfer=False, finetune=False  -> Invalid.

    The chaining strategy is controlled by ``(use_true_labels, use_oof)``
    (evaluated in order; ``use_true_labels`` takes precedence):

      use_true_labels=True            -> Use measured η(10) as chaining feature
                                         at training time (oracle chain).
      use_oof=True                    -> Out-of-fold η(10) predictions (honest).
      both False                      -> In-sample η(10) predictions (paper-faithful).
    """
    id: str
    transfer: bool
    chained: bool
    use_oof: bool
    use_true_labels: bool
    group_aware_split: bool
    finetune: bool

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_Variant":
        return cls(
            id=str(d["id"]),
            transfer=bool(d.get("transfer", True)),
            chained=bool(d.get("chained", True)),
            use_oof=bool(d.get("use_oof", False)),
            use_true_labels=bool(d.get("use_true_labels", False)),
            group_aware_split=bool(d.get("group_aware_split", False)),
            finetune=bool(d.get("finetune", True)),
        )


def _active_variants(cfg: Config) -> List[_Variant]:
    raw: List[Dict[str, Any]] = list(getattr(cfg.tl, "variants", []))
    if not raw:
        # No explicit variants — fall back to the paper-faithful single setup.
        raw = [{"id": "tl_paper_chained", "transfer": True,
                "chained": True, "use_oof": False,
                "group_aware_split": False, "finetune": True}]
    variants = [_Variant.from_dict(v) for v in raw if v.get("active", True)]
    for v in variants:
        if not v.transfer and not v.finetune:
            raise ValueError(
                f"Variant {v.id!r}: transfer=False with finetune=False is "
                f"invalid — there is no training data left."
            )
    return variants


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_targets(
    df: pd.DataFrame,
    target_cols: Sequence[str],
) -> Dict[str, np.ndarray]:
    return {tgt: df[tgt].to_numpy(dtype=float).ravel() for tgt in target_cols}


# ---------------------------------------------------------------------------
# Target transforms
# ---------------------------------------------------------------------------


def _apply_target_transform(
    y_dict: Dict[str, np.ndarray],
    mode: str,
) -> Dict[str, np.ndarray]:
    """Forward-transform every target array in a dict.

    Currently supported modes:
      ``"none"``               -- identity (returns input unchanged)
      ``"log1p_nonnegative"``  -- log1p applied to clipped non-negative values

    Viscosities are physically non-negative; the clip protects against any
    tiny negative noise that would otherwise produce NaNs.
    """
    if mode == "none":
        return y_dict
    if mode == "log1p_nonnegative":
        return {
            tgt: np.log1p(np.maximum(np.asarray(arr, float), 0.0))
            for tgt, arr in y_dict.items()
        }
    raise ValueError(
        f"Unknown target transform {mode!r}; expected 'none' or 'log1p_nonnegative'"
    )


def _invert_target_transform(
    y_dict: Dict[str, np.ndarray],
    mode: str,
) -> Dict[str, np.ndarray]:
    """Inverse-transform predictions back into the original target units.

    ``expm1`` for ``log1p_nonnegative``; identity otherwise.  Negative values
    after expm1 should be impossible for non-negative input but are clipped
    to zero defensively so R² / RMSE never gets NaNs from the inverse.
    """
    if mode == "none":
        return y_dict
    if mode == "log1p_nonnegative":
        return {
            tgt: np.maximum(np.expm1(np.asarray(arr, float)), 0.0)
            for tgt, arr in y_dict.items()
        }
    raise ValueError(
        f"Unknown target transform {mode!r}; expected 'none' or 'log1p_nonnegative'"
    )


def _encode_inputs_with_train_categories(
    pretrain: pd.DataFrame,
    target_train: pd.DataFrame,
    target_test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """One-hot encode all three frames with categories fitted on (pretrain ∪ target_train).

    Test categories that don't appear in the fit set are silently dropped
    (``pd.Categorical`` returns NaN for unseen categories, which becomes 0
    after ``pd.get_dummies(..., dummy_na=False)``).  This is the standard
    safe behaviour for held-out splits.
    """
    fit_basis = pd.concat([pretrain[feature_cols], target_train[feature_cols]],
                          axis=0, ignore_index=True)
    _, fit_categories = one_hot_encode_inputs(fit_basis, feature_cols)
    Xs_df, _ = one_hot_encode_inputs(pretrain[feature_cols], feature_cols,
                                     fit_categories=fit_categories)
    Xt_train_df, _ = one_hot_encode_inputs(target_train[feature_cols], feature_cols,
                                           fit_categories=fit_categories)
    Xt_test_df, _ = one_hot_encode_inputs(target_test[feature_cols], feature_cols,
                                          fit_categories=fit_categories)
    encoded_cols = list(Xs_df.columns)
    return (
        Xs_df.to_numpy(dtype=float),
        Xt_train_df.to_numpy(dtype=float),
        Xt_test_df.to_numpy(dtype=float),
        encoded_cols,
    )


# ---------------------------------------------------------------------------
# Per-fold runner
# ---------------------------------------------------------------------------


def _fit_predict_for_variant(
    variant: _Variant,
    learner_name: str,
    *,
    X_source: np.ndarray,
    y_source_dict: Dict[str, np.ndarray],
    X_target_train: np.ndarray,
    y_target_train_dict: Dict[str, np.ndarray],
    X_test: np.ndarray,
    seed: int,
    target_cols: Sequence[str],
    encoded_feature_cols: Sequence[str],
    eta10_target: str,
    eta1_target: str,
    eta100_target: str,
    learner_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Fit one variant on the requested data and return per-target test preds.

    Routing matrix:
      transfer=True,  finetune=True   -> source = Batch_1, fine-tune on B2 fold
      transfer=True,  finetune=False  -> source = Batch_1, no fine-tune
                                          (paper's "Train B1 → Predict B2 (no TL)")
      transfer=False, finetune=True   -> source = B2 fold, no fine-tune
                                          (Batch-2-only training)
    """
    if variant.transfer and variant.finetune:
        X_src, y_src = X_source, y_source_dict
        X_tgt, y_tgt = X_target_train, y_target_train_dict
    elif variant.transfer and not variant.finetune:
        # Paper's true no-TL baseline: never see Batch_2 at training time.
        X_src, y_src = X_source, y_source_dict
        X_tgt, y_tgt = None, None
    else:
        # transfer=False, finetune=True (no transfer, train on target only).
        X_src, y_src = X_target_train, y_target_train_dict
        X_tgt, y_tgt = None, None

    if variant.chained:
        fitr = fit_chained(
            learner_name,
            X_src,
            y_src,
            X_tgt if X_tgt is not None else np.empty((0, X_src.shape[1])),
            y_tgt if y_tgt is not None else {tgt: np.empty(0) for tgt in target_cols},
            eta10_target=eta10_target,
            eta1_target=eta1_target,
            eta100_target=eta100_target,
            feature_cols=list(encoded_feature_cols),
            use_true_labels=variant.use_true_labels,
            use_oof=variant.use_oof,
            seed=seed,
            learner_kwargs=learner_kwargs,
        )
        preds = predict_chained(fitr, X_test)
    else:
        fitr = fit_direct(
            learner_name,
            X_src,
            y_src,
            X_tgt if X_tgt is not None else np.empty((0, X_src.shape[1])),
            y_tgt if y_tgt is not None else {tgt: np.empty(0) for tgt in target_cols},
            target_cols=list(target_cols),
            feature_cols=list(encoded_feature_cols),
            seed=seed,
            learner_kwargs=learner_kwargs,
        )
        preds = predict_direct(fitr, X_test)
    return preds


# ---------------------------------------------------------------------------
# Section entry point
# ---------------------------------------------------------------------------


def run_tl_baseline(
    cfg: Config,
    *,
    run_id: Optional[str] = None,
) -> SectionResult:
    """Run the transfer-learning replication and return a SectionResult.

    The signature deliberately omits ``physics_datasets`` because the TL
    pipeline does not consume the fitted-physics datasets — it works
    directly on the raw composition columns and viscosity measurements,
    matching the published protocol.

    The ``run_id`` argument is accepted for parity with other section
    runners in the project, but currently unused.  It is kept so future
    versions can drop curve-overlay PNGs into the run directory at
    execution time without changing the runner's call signature.
    """
    tl_cfg = cfg.tl  # type: ignore[attr-defined]
    seed = int(tl_cfg.seed if tl_cfg.seed is not None else cfg.seed)

    # Step 1: load and partition by Source_Batch.
    split = prepare_tl_split(
        cfg,
        source_batch=tl_cfg.source_batch,
        target_batch=tl_cfg.target_batch,
    )
    print(
        f"\n{'='*70}\n"
        f"Transfer-Learning replication — paper protocol (Phase 1) + audit (Phase 2)\n"
        f"{'='*70}\n"
        f"  Pretrain batch ({tl_cfg.source_batch}): {len(split.pretrain)} rows\n"
        f"  Target batch   ({tl_cfg.target_batch}): {len(split.target)} rows"
    )

    target_cols = list(split.target_cols)  # paper viscosity columns
    eta1_col, eta10_col, eta100_col = target_cols  # ordering set in data.PAPER_VISCOSITY_COLS

    learner_kwargs_per: Dict[str, Dict[str, Any]] = dict(getattr(tl_cfg, "lightgbm", {}) or {})
    learner_kwargs_for: Dict[str, Dict[str, Any]] = {"lightgbm": learner_kwargs_per}

    # Step 2: outer 80/20 (or group-aware) split on the target batch.
    per_target_rows: List[dict] = []
    model_summary_rows: List[dict] = []
    all_predictions: List[pd.DataFrame] = []
    fold_rows: List[FoldRow] = []
    conformal_rows: List[dict] = []

    kfold_scope = str(getattr(tl_cfg, "kfold_scope", "full_target"))
    target_transform = str(getattr(tl_cfg, "target_transform", "none"))
    print(f"  K-fold scope     : {kfold_scope}")
    print(f"  Target transform : {target_transform}")

    for variant in _active_variants(cfg):
        print(f"\n→ Variant {variant.id}  "
              f"(transfer={variant.transfer}, finetune={variant.finetune}, "
              f"chained={variant.chained}, use_oof={variant.use_oof}, "
              f"use_true_labels={variant.use_true_labels}, "
              f"group_aware={variant.group_aware_split})")

        # K-fold scope decides which rows the inner CV splits.
        #   "full_target"     -- K-fold across all rows of Batch_2 (paper Sec 3.1).
        #   "train_partition" -- outer 80/20 then K-fold inside the 80% portion
        #                        (the original Phase-1 default; kept for back-compat).
        if kfold_scope == "full_target":
            target_train = split.target.reset_index(drop=True)
            target_test = split.target.iloc[0:0].reset_index(drop=True)  # unused
            print(f"  K-fold scope     : full Batch_2 ({len(target_train)} rows)")
        else:
            target_train, target_test = split_target_batch(
                split.target,
                test_fraction=float(tl_cfg.test_fraction),
                seed=seed,
                group_aware=variant.group_aware_split,
                group_col=split.group_col,
            )
            print(f"  Outer 80/20 split: train={len(target_train)}, test={len(target_test)}")

        # k-fold inside whichever scope was selected (paper: k=5 by default).
        kf = KFold(
            n_splits=int(tl_cfg.k_folds),
            shuffle=True,
            random_state=seed,
        )

        for learner_name in tl_cfg.models:
            if learner_name not in _VALID_LEARNERS:
                raise ValueError(
                    f"Unknown TL learner {learner_name!r}; valid: {_VALID_LEARNERS}"
                )

            learner_kwargs = learner_kwargs_for.get(learner_name)

            # The outer test set is held aside for the final report;
            # CV folds happen on the inner target_train portion.
            for fold_idx, (inner_train_idx, inner_test_idx) in enumerate(
                kf.split(np.arange(len(target_train)))
            ):
                tgt_train_fold = target_train.iloc[inner_train_idx].reset_index(drop=True)
                tgt_test_fold = target_train.iloc[inner_test_idx].reset_index(drop=True)

                Xs_arr, Xtt_arr, Xte_arr, encoded_cols = _encode_inputs_with_train_categories(
                    split.pretrain,
                    tgt_train_fold,
                    tgt_test_fold,
                    list(split.feature_cols),
                )
                ys = _encode_targets(split.pretrain, target_cols)
                ytt = _encode_targets(tgt_train_fold, target_cols)
                yte = _encode_targets(tgt_test_fold, target_cols)

                # Fit in transformed target space (e.g. log1p) when requested,
                # then invert before scoring so R²/MAE/RMSE are in Pa·s units
                # and comparable across runs.  ``yte`` (ground truth) stays in
                # original units throughout.
                ys_fit = _apply_target_transform(ys, target_transform)
                ytt_fit = _apply_target_transform(ytt, target_transform)

                preds_fit = _fit_predict_for_variant(
                    variant,
                    learner_name,
                    X_source=Xs_arr,
                    y_source_dict=ys_fit,
                    X_target_train=Xtt_arr,
                    y_target_train_dict=ytt_fit,
                    X_test=Xte_arr,
                    seed=seed + fold_idx,
                    target_cols=target_cols,
                    encoded_feature_cols=encoded_cols,
                    eta10_target=eta10_col,
                    eta1_target=eta1_col,
                    eta100_target=eta100_col,
                    learner_kwargs=learner_kwargs,
                )
                preds = _invert_target_transform(preds_fit, target_transform)

                # Per-fold scoring and conformal calibration.
                for tgt in target_cols:
                    s = score_predictions(yte[tgt], preds[tgt])
                    fold_rows.append(FoldRow(
                        fold=fold_idx,
                        learner=learner_name,
                        variant=variant.id,
                        target=tgt,
                        r2=s["r2"],
                        mae=s["mae"],
                        rmse=s["rmse"],
                        n=s["n"],
                    ))

                    # Split-conformal: use this fold's training portion as the
                    # combined proper-train + calibration source.  Predictions
                    # on the calibration rows are produced by re-fitting on the
                    # proper-train split only.  We approximate that by
                    # reusing the variant's prediction routine on a
                    # proper/calibration split inside ``inner_train_idx``.
                    if tl_cfg.conformal.alphas:
                        proper_idx, calib_idx = split_proper_calib(
                            len(tgt_train_fold),
                            calib_fraction=float(tl_cfg.conformal.calib_fraction),
                            seed=seed + fold_idx,
                        )
                        proper_train = tgt_train_fold.iloc[proper_idx].reset_index(drop=True)
                        calibration = tgt_train_fold.iloc[calib_idx].reset_index(drop=True)
                        Xs2, Xpt, Xca, _ = _encode_inputs_with_train_categories(
                            split.pretrain,
                            proper_train,
                            calibration,
                            list(split.feature_cols),
                        )
                        ys2 = _encode_targets(split.pretrain, target_cols)
                        ypt = _encode_targets(proper_train, target_cols)
                        yca = _encode_targets(calibration, target_cols)

                        ys2_fit = _apply_target_transform(ys2, target_transform)
                        ypt_fit = _apply_target_transform(ypt, target_transform)

                        cal_preds_fit = _fit_predict_for_variant(
                            variant,
                            learner_name,
                            X_source=Xs2,
                            y_source_dict=ys2_fit,
                            X_target_train=Xpt,
                            y_target_train_dict=ypt_fit,
                            X_test=Xca,
                            seed=seed + 1000 + fold_idx,
                            target_cols=target_cols,
                            encoded_feature_cols=encoded_cols,
                            eta10_target=eta10_col,
                            eta1_target=eta1_col,
                            eta100_target=eta100_col,
                            learner_kwargs=learner_kwargs,
                        )
                        cal_preds = _invert_target_transform(cal_preds_fit, target_transform)

                        for alpha in tl_cfg.conformal.alphas:
                            q = split_conformal_quantile(yca[tgt], cal_preds[tgt], alpha)
                            lo, hi = conformal_intervals(preds[tgt], q)
                            cov, width, n = conformal_coverage(yte[tgt], lo, hi)
                            conformal_rows.append({
                                "learner": learner_name,
                                "variant": variant.id,
                                "target": tgt,
                                "fold": fold_idx,
                                "alpha": float(alpha),
                                "coverage": cov,
                                "mean_width": width,
                                "n": n,
                            })

                # Long-format predictions for this fold.
                all_predictions.append(build_long_predictions(
                    y_true=yte,
                    y_pred=preds,
                    learner=learner_name,
                    variant=variant.id,
                    fold=fold_idx,
                ))

    # Step 3: aggregate per-fold rows.
    per_target_df = aggregate_folds(
        fold_rows,
        group_by=("learner", "variant", "target"),
    )
    model_summary_df = aggregate_folds(
        fold_rows,
        group_by=("learner", "variant"),
    )

    # Conformal coverage roll-up onto per-target rows.
    if conformal_rows:
        conf_df = pd.DataFrame(conformal_rows)
        roll = (
            conf_df.groupby(["learner", "variant", "target", "alpha"], sort=False)
            .agg(coverage=("coverage", "mean"),
                 mean_width=("mean_width", "mean"))
            .reset_index()
        )
        # Pivot so each α produces its own pair of columns.
        wide = roll.pivot_table(
            index=["learner", "variant", "target"],
            columns="alpha",
            values=["coverage", "mean_width"],
        )
        wide.columns = [
            f"{metric}_{int(round((1 - alpha) * 100))}"
            for metric, alpha in wide.columns
        ]
        wide = wide.reset_index()
        per_target_df = per_target_df.merge(
            wide, how="left", on=["learner", "variant", "target"]
        )

    # Add a unifying ``model_name`` column so downstream code that expects it
    # (built around the physics-model registry) keeps working.  TL uses the
    # learner name as the model identifier.
    per_target_df["model_name"] = per_target_df["learner"]
    per_target_df["model_key"] = per_target_df["learner"]
    model_summary_df["model_name"] = model_summary_df["learner"]
    model_summary_df["model_key"] = model_summary_df["learner"]
    if "test_r2" not in model_summary_df.columns:
        # Provide convenience aliases matching what results.best() looks for.
        model_summary_df["test_r2"] = model_summary_df["r2_mean"]
        model_summary_df["test_rmse"] = model_summary_df["rmse_mean"]

    fold_summary_df = pd.DataFrame([fr.__dict__ for fr in fold_rows]) if fold_rows else None
    if conformal_rows and fold_summary_df is not None:
        # Annotate fold rows with conformal columns when available, for
        # interactive inspection.  Aggregations have already been written
        # to per_target_df above.
        pass

    predictions_df = (
        pd.concat(all_predictions, axis=0, ignore_index=True) if all_predictions else None
    )

    return SectionResult(
        name="tl_baseline",
        model_summary=model_summary_df,
        per_target=per_target_df,
        predictions=predictions_df,
        fold_summary=fold_summary_df,
        curve_summary=None,
    )
