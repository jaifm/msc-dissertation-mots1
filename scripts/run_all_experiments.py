"""Run all outstanding experiments and generate full plot suites.

Executes in sequence:
  exp_017_shap_pruned            — XGBoost optimised baseline (fresh run + plots)
  exp_021_gpr_matern25_linear    — GPR best config (fresh run + plots)
  exp_024_loo_optimised          — LOO row/mix/batch (first run)
  exp_025_xgb_gpr_comparison     — Direct XGBoost vs GPR on identical feature set

After each experiment, physics_datasets are re-built so that the full plot
suite (including curve overlay) can be generated in the run directory.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py 2>&1 | tee run_log.txt
"""
from __future__ import annotations

import sys
import time
import traceback

from src.config import load_config
from src.data.splits import load_splits
from src.physics.datasets import build_all_physics_datasets
from src.experiments.runner import run_experiment
from src.viz import make_all_plots

EXPERIMENTS = [
    "exp_017_shap_pruned",
    "exp_021_gpr_matern25_linear",
    "exp_024_loo_optimised",
    "exp_025_xgb_gpr_comparison",
]


def _banner(msg: str, width: int = 72) -> None:
    print(f"\n{'='*width}")
    print(f"  {msg}")
    print(f"{'='*width}")


def run_one(exp_id: str) -> bool:
    """Run a single experiment and generate all plots.  Returns True on success."""
    _banner(f"RUNNING: {exp_id}")
    t0 = time.perf_counter()
    try:
        cfg = load_config(exp_id)
        results = run_experiment(cfg, persist=True)

        if results.results_dir is None:
            print(f"  [warn] No results_dir — skipping plots for {exp_id}")
            return True

        # Re-build physics datasets so curve-overlay plots can be generated.
        # This is the only step that needs raw data in memory; all other plots
        # can be generated from the saved CSVs, but we do it all at once here.
        print(f"\n  Generating plots → {results.results_dir}")
        train_df, test_df = load_splits(cfg)
        physics_datasets = build_all_physics_datasets(train_df, test_df, cfg)
        plot_paths = make_all_plots(
            results,
            results.results_dir,
            physics_datasets=physics_datasets,
            cfg=cfg,
        )
        print(f"  Saved {len(plot_paths)} plot(s)")

        elapsed = time.perf_counter() - t0
        mins, secs = divmod(int(elapsed), 60)
        print(f"\n  ✓ {exp_id} complete in {mins}m {secs:02d}s")
        return True

    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"\n  ✗ {exp_id} FAILED after {elapsed:.0f}s")
        traceback.print_exc()
        return False


def main() -> None:
    overall_t0 = time.perf_counter()
    print(f"Starting {len(EXPERIMENTS)} experiments …")

    successes, failures = [], []
    for exp_id in EXPERIMENTS:
        ok = run_one(exp_id)
        (successes if ok else failures).append(exp_id)

    total = time.perf_counter() - overall_t0
    hrs, rem = divmod(int(total), 3600)
    mins, secs = divmod(rem, 60)

    _banner("SUMMARY")
    print(f"  Total time : {hrs}h {mins:02d}m {secs:02d}s")
    print(f"  Succeeded  : {len(successes)}")
    for s in successes:
        print(f"    ✓ {s}")
    if failures:
        print(f"  Failed     : {len(failures)}")
        for f in failures:
            print(f"    ✗ {f}")
    print()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
