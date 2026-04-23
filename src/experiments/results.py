"""Typed experiment result containers and disk persistence.

OVERVIEW
--------
Every experiment section (baseline, sweep, LOO) produces a ``SectionResult``
containing three DataFrames.  An experiment run can include multiple sections,
all bundled into a single ``ExperimentResults`` object.

This module also handles writing results to disk and maintaining the shared
experiment manifest, which allows the notebook's comparison cell to list all
completed runs without re-running any computations.

DATA STRUCTURES
---------------
SectionResult
    The atomic unit of experiment output for one section:
    - ``model_summary``: one row per model (averaged over targets), suitable
      for ranking models by test R².
    - ``per_target``: one row per (model, target) combination, allowing
      diagnosis of which physics parameters are easiest/hardest to predict.
    - ``predictions``: optional; row-level (y_true, y_pred) for scatter plots
      and residual analysis.  Not all sections collect this (see sweep.py).

ExperimentResults
    A dict-like container holding one SectionResult per section name, plus
    the config experiment ID, a unique run ID, and a UTC timestamp.

UNIQUE RUN IDS
--------------
Every call to ``run_experiment`` produces a distinct ``run_id`` of the form
``{experiment_id}_{YYYYMMDD_HHMMSS}`` (e.g. ``exp_001_baseline_20260420_143022``).
Results are written to ``experiments_root / run_id``, so re-running the same
experiment config never overwrites previous results — each run gets its own
subdirectory.

The manifest accumulates one entry per run (all runs are preserved) and is
sorted most-recent-first.  The notebook's comparison cell therefore shows the
full history of every config that has been run.

PERSISTENCE LAYOUT
------------------
Results are saved to ``experiments_root / run_id``:

  <section>_model_summary.csv    -- model_summary DataFrame
  <section>_per_target.csv       -- per_target DataFrame
  <section>_predictions.csv      -- predictions DataFrame (if present)
  config_snapshot.json           -- fully resolved config at run time
  metadata.json                  -- summary metadata (best model, files, etc.)

The config snapshot is saved alongside the results so that any future
re-analysis knows exactly which hyperparameters, feature settings, and data
paths were active during that specific run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import Config


def _make_run_id(experiment_id: str) -> str:
    """Generate a unique run ID by appending a compact UTC timestamp.

    Format: ``{experiment_id}_{YYYYMMDD_HHMMSS}``

    The timestamp is in UTC so run IDs sort correctly regardless of the
    local timezone of the machine running the experiment.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{experiment_id}_{ts}"


@dataclass
class SectionResult:
    """Results from a single section of an experiment run.

    Attributes
    ----------
    name:
        Section identifier, e.g. ``'baseline'``, ``'sweep'``, ``'loo'``.
    model_summary:
        One row per model (and variant, for the sweep section), with metrics
        averaged over all targets that were evaluated.  Primary table for
        ranking and comparing models.
    per_target:
        One row per (model, target) combination.  Provides a finer-grained
        view of which physics parameters the model predicts well or poorly.
    predictions:
        Optional DataFrame with columns ``[y_true, y_pred, target, model_name,
        ...]``.  Used for scatter plots and residual analysis.  Not all
        sections populate this field (the sweep section skips it to save
        storage).
    """
    name: str
    model_summary: pd.DataFrame
    per_target: pd.DataFrame
    predictions: Optional[pd.DataFrame] = None

    def sorted_by_r2(self) -> pd.DataFrame:
        """Return model_summary sorted by test R² descending.

        Convenience method used by the notebook inspection cell and by the
        plotting functions to rank models without repeating the sort logic.
        """
        return self.model_summary.sort_values("test_r2", ascending=False).reset_index(drop=True)


@dataclass
class ExperimentResults:
    """All section outputs from a single experiment run.

    Acts as a dict of SectionResults keyed by section name, with a few
    convenience methods for querying across sections.

    Attributes
    ----------
    experiment_id:
        The config ID string (e.g. ``'exp_001_baseline'``).  Shared across
        all runs of the same config.
    run_id:
        Unique identifier for this specific run, formed as
        ``{experiment_id}_{YYYYMMDD_HHMMSS}``.  Used as the output directory
        name so multiple runs of the same config never collide on disk.
    sections:
        Dict mapping section name → SectionResult.
    created_at_utc:
        ISO-8601 UTC timestamp set at object creation time.
    results_dir:
        Set by ``save()`` to the directory where this run's files were written.
        None until ``save()`` is called (i.e. when ``persist=False``).
    """
    experiment_id: str
    run_id: str = field(default="")
    sections: Dict[str, SectionResult] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results_dir: Optional[Path] = field(default=None)
    duration_seconds: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        # Generate run_id at construction time if not supplied explicitly.
        # This ensures the timestamp reflects when the run started, not when
        # save() was called.
        if not self.run_id:
            self.run_id = _make_run_id(self.experiment_id)

    def add(self, section: SectionResult) -> None:
        """Register a SectionResult under its ``name`` key."""
        self.sections[section.name] = section

    def best(self, section_name: str = "baseline") -> Optional[dict]:
        """Return the top-ranked model from a given section, or None.

        Falls back gracefully if the section does not exist or has no rows,
        which can happen when an experiment only runs the LOO section and
        ``section_name='baseline'`` is requested.

        Parameters
        ----------
        section_name:
            Which section to query.  Defaults to ``'baseline'``.

        Returns
        -------
        dict with keys ``model_name``, ``test_r2``, ``test_rmse``, or None.
        """
        section = self.sections.get(section_name)
        if section is None or section.model_summary.empty:
            return None
        best = section.sorted_by_r2().iloc[0]
        return {
            "model_name": best["model_name"],
            "test_r2": float(best["test_r2"]),
            "test_rmse": float(best["test_rmse"]),
        }


# ---------------------------------------------------------------------------
# Disk persistence
# ---------------------------------------------------------------------------


def save(cfg: Config, results: ExperimentResults) -> Path:
    """Write all section DataFrames, metadata, and config snapshot to disk.

    Each run is written to its own subdirectory named after ``results.run_id``
    (``experiments_root / run_id``), so re-running the same experiment config
    never overwrites previous results.

    Also sets ``results.results_dir`` to the output path so the caller can
    pass it directly to ``make_all_plots`` without re-deriving it.

    Parameters
    ----------
    cfg:
        Experiment configuration (provides the experiments root path).
    results:
        The completed experiment results to persist.

    Returns
    -------
    Path
        The run-specific results directory (``experiments_root / run_id``).
    """
    # Use run_id as the directory name, not experiment_id, so each run is
    # isolated even when the same config is executed multiple times.
    out_dir = cfg.experiments_root_path / results.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results.results_dir = out_dir

    saved_files: List[str] = []
    for section in results.sections.values():
        base = f"{section.name}_"
        ms_path = out_dir / f"{base}model_summary.csv"
        pt_path = out_dir / f"{base}per_target.csv"
        section.model_summary.to_csv(ms_path, index=False)
        section.per_target.to_csv(pt_path, index=False)
        saved_files += [ms_path.name, pt_path.name]
        if section.predictions is not None and not section.predictions.empty:
            pp = out_dir / f"{base}predictions.csv"
            section.predictions.to_csv(pp, index=False)
            saved_files.append(pp.name)

    # Save the fully resolved config at run time so results are self-contained.
    # This snapshot includes any overrides from the experiment YAML, so the
    # exact settings used can always be recovered without re-running the code.
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str)
    )

    # Per-run metadata record: a compact summary that the manifest and the
    # notebook comparison cell can display without loading the full CSVs.
    # 'best' falls through the three section names in priority order.
    best = results.best("baseline") or results.best("sweep") or results.best("loo")
    metadata = {
        "run_id": results.run_id,
        "experiment_id": results.experiment_id,
        "description": cfg.description,
        "hypothesis": cfg.hypothesis,
        "created_at_utc": results.created_at_utc,
        "duration_seconds": results.duration_seconds,
        "sections_run": list(results.sections.keys()),
        "best": best,
        "saved_files": saved_files,
        # Free-text annotation field.  Empty at save time; populated later via
        # annotate_run() without re-running the experiment.
        "notes": "",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    _update_manifest(cfg, metadata)
    return out_dir


def _update_manifest(cfg: Config, metadata: dict) -> None:
    """Append this run's record to the shared experiment manifest.

    Every run is preserved — the manifest accumulates the full history of all
    runs across all experiment configs, sorted most-recent-first.  Deduplication
    is on ``run_id`` (which is unique per run), not ``experiment_id``, so
    re-running the same config produces a new entry rather than replacing the
    old one.
    """
    manifest_path = cfg.experiments_root_path / "experiment_manifest.json"
    manifest = {"runs": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            # If the manifest is corrupt (e.g. from a partial write), start fresh
            # rather than propagating the error or silently dropping the new run.
            pass
    manifest.setdefault("runs", [])
    # Deduplicate on run_id only: if the exact same run_id is already in the
    # manifest (e.g. save() called twice in one session), replace it.
    # Different runs of the same experiment_id are always kept.
    manifest["runs"] = [
        r for r in manifest["runs"] if r.get("run_id") != metadata["run_id"]
    ]
    manifest["runs"].insert(0, metadata)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def annotate_run(run_id: str, note: str, experiments_root: Path) -> None:
    """Add or replace a free-text note on an existing manifest entry.

    Notes are stored directly in the manifest JSON so they persist across
    sessions and are visible in the notebook comparison table without
    re-running the experiment.  The per-run ``metadata.json`` file is also
    updated so the annotation is self-contained within the run directory.

    Parameters
    ----------
    run_id:
        The unique run identifier (e.g. ``'exp_001_baseline_20260421_143022'``).
        Shown in the 'Run ID' column of the notebook comparison table.
    note:
        Free-text annotation.  Can be multi-line.  Replaces any existing note
        for this run — to append, read the current note first and concatenate.
    experiments_root:
        Path to the top-level experiments directory (contains the manifest and
        all run subdirectories).  Typically ``cfg.experiments_root_path``.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    KeyError
        If ``run_id`` is not found in the manifest.

    Examples
    --------
    >>> annotate_run(
    ...     results.run_id,
    ...     "Power Law best; CY5 overfit badly. Log transform helped eta_0.",
    ...     cfg.experiments_root_path,
    ... )
    """
    manifest_path = experiments_root / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    matched = False
    for run in manifest["runs"]:
        if run.get("run_id") == run_id:
            run["notes"] = note
            matched = True
            break
    if not matched:
        raise KeyError(
            f"run_id {run_id!r} not found in manifest. "
            "Check the 'run_id' column in the comparison table for the correct ID."
        )

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    # Mirror the note into the per-run metadata.json so the run directory
    # remains self-contained and the note survives if the manifest is rebuilt.
    run_metadata_path = experiments_root / run_id / "metadata.json"
    if run_metadata_path.exists():
        run_meta = json.loads(run_metadata_path.read_text())
        run_meta["notes"] = note
        run_metadata_path.write_text(json.dumps(run_meta, indent=2, default=str))
