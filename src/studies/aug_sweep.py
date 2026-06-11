"""Sweep n_aug_points to map diminishing returns of augmentation volume.

Loads the base experiment config (all oracles, log1p target, equal weights)
and re-runs the physics_augmented section for each point count.  Results are
persisted under ids of the form ``<base>_aug_sweep_<N>pts`` so they appear
in the manifest and can be loaded for comparison.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List

import pandas as pd

DEFAULTS: Dict[str, Any] = {
    "base_experiment": "exp_030_pa_log_target",
    "aug_points": [10, 20, 50, 75, 100, 150, 300, 500],
    # Persist each sweep run to the experiment manifest.
    "persist": True,
}


def _banner(msg: str, width: int = 72) -> None:
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


def _run_one(n_aug: int, params: Dict[str, Any]) -> dict:
    from ..config import load_config
    from ..experiments.runner import run_experiment

    base_exp = params["base_experiment"]
    label = f"{base_exp}_aug_sweep_{n_aug:04d}pts"
    _banner(f"n_aug_points={n_aug}  →  {label}")
    t0 = time.perf_counter()
    try:
        cfg = load_config(base_exp)
        cfg.physics_augmented.n_aug_points = int(n_aug)
        cfg.id = label

        results = run_experiment(cfg, persist=bool(params["persist"]))

        elapsed = time.perf_counter() - t0
        section = results.sections.get("physics_augmented")
        record = {"n_aug": n_aug, "ok": True, "seconds": round(elapsed, 1)}
        if section is not None and not section.model_summary.empty:
            best = section.model_summary.sort_values(
                "test_r2_log", ascending=False
            ).iloc[0]
            record.update({
                "best_variant": best["variant"],
                "r2_raw": float(best["test_r2"]),
                "r2_log": float(best["test_r2_log"]),
                "rmse": float(best["test_rmse"]),
            })
            print(f"\n  best: {best['variant']}  "
                  f"R2(raw)={best['test_r2']:.3f}  "
                  f"R2(log)={best['test_r2_log']:.3f}  "
                  f"RMSE={best['test_rmse']:.3f}")
        mins, secs = divmod(int(elapsed), 60)
        print(f"  done in {mins}m {secs:02d}s")
        return record

    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"\n  FAILED after {elapsed:.0f}s")
        traceback.print_exc()
        return {"n_aug": n_aug, "ok": False, "seconds": round(elapsed, 1)}


def run(params: Dict[str, Any]) -> pd.DataFrame:
    overall_t0 = time.perf_counter()
    aug_points: List[int] = list(params["aug_points"])
    print(f"Aug-points sweep: {aug_points}")

    records = [_run_one(n, params) for n in aug_points]

    total = time.perf_counter() - overall_t0
    hrs, rem = divmod(int(total), 3600)
    mins, secs = divmod(rem, 60)

    _banner("SUMMARY")
    print(f"  Total time : {hrs}h {mins:02d}m {secs:02d}s")
    print(f"  Succeeded  : {[r['n_aug'] for r in records if r['ok']]}")
    failures = [r["n_aug"] for r in records if not r["ok"]]
    if failures:
        print(f"  Failed     : {failures}")
    return pd.DataFrame(records)
