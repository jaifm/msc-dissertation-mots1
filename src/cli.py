"""Command-line interface for running and listing experiments.

OVERVIEW
--------
This module provides a thin CLI wrapper around ``run_experiment`` so that
experiments can be triggered from a terminal without opening a notebook.
This is useful for running long experiments in the background, scheduling
them with a job system, or running them on a remote server where a browser
is not available.

COMMANDS
--------
``python -m src.cli run --config <experiment_id>``
    Load the named config and run the experiment.  Results are saved to
    ``cfg.results_dir`` unless ``--no-save`` is passed.

    Example:
        python -m src.cli run --config exp_001_baseline
        python -m src.cli run --config exp_004_trials_20_raw --no-save

``python -m src.cli list``
    Print all available experiment config names (YAML stems under
    ``configs/experiments/``) one per line.  Useful for scripting.

``python -m src.cli study --name <study_name>``
    Run a registered analysis study (see ``src/studies``).  Parameters come
    from the study module's documented defaults overridden by
    ``configs/studies/<study_name>.yaml`` — there are deliberately no
    parameter flags, keeping studies as config-first as experiments.

    Example:
        python -m src.cli study --name shear_holdout

``python -m src.cli list-studies``
    Print all registered study names one per line.

DESIGN NOTE
-----------
The CLI is kept intentionally minimal.  It does no logging configuration,
no parallelism, and no result inspection.  All of that is handled by the
notebook or by reading the saved CSV/JSON files directly.  The CLI's only
job is to translate a command-line string into a ``run_experiment`` call.
"""
from __future__ import annotations

import argparse
import sys

from .config import list_experiments, load_config
from .experiments.runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate command.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when None,
        which is the standard behaviour for CLI entry points.  Passing an
        explicit list is useful for testing.

    Returns
    -------
    int
        Exit code: 0 for success, 1 if the command was unrecognised.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Run or list slurry rheology prediction experiments.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- 'run' subcommand ---------------------------------------------------
    run_p = sub.add_parser("run", help="Run an experiment by config name.")
    run_p.add_argument(
        "--config",
        required=True,
        help="Experiment id (YAML stem under configs/experiments/), e.g. exp_001_baseline.",
    )
    run_p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing results to disk.  Useful for quick checks.",
    )

    # --- 'list' subcommand --------------------------------------------------
    sub.add_parser("list", help="List all available experiment config names.")

    # --- 'study' subcommand ---------------------------------------------------
    study_p = sub.add_parser("study", help="Run a registered analysis study.")
    study_p.add_argument(
        "--name",
        required=True,
        help="Study name (see 'list-studies'), e.g. shear_holdout.",
    )

    # --- 'list-studies' subcommand --------------------------------------------
    sub.add_parser("list-studies", help="List all registered study names.")

    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name in list_experiments():
            print(name)
        return 0

    if args.cmd == "list-studies":
        from .studies import list_studies
        for name in list_studies():
            print(name)
        return 0

    if args.cmd == "study":
        from .studies import run_study
        run_study(args.name)
        return 0

    if args.cmd == "run":
        cfg = load_config(args.config)
        run_experiment(cfg, persist=not args.no_save)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
