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
    the experiment ID and a UTC timestamp of when the run was created.

PERSISTENCE LAYOUT
------------------
Results are saved to ``cfg.results_dir`` (= experiments_root / experiment_id):

  <section>_model_summary.csv    -- model_summary DataFrame
  <section>_per_target.csv       -- per_target DataFrame
  <section>_predictions.csv      -- predictions DataFrame (if present)
  config_snapshot.json           -- fully resolved config at run time
  metadata.json                  -- summary metadata (best model, files, etc.)

The config snapshot is saved alongside the results so that any future
re-analysis knows exactly which hyperparameters, feature settings, and data
paths were active during the run.

EXPERIMENT MANIFEST
-------------------
``_update_manifest`` appends a metadata record to a shared
``experiment_manifest.json`` at the root of the experiments directory.  The
most recent run for each experiment ID is kept at position 0; older runs for
the same ID are replaced.  This allows the comparison notebook cell to load
all experiments without scanning individual subdirectories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import Config


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
        The config ID string (e.g. ``'exp_001_baseline'``).
    sections:
        Dict mapping section name → SectionResult.
    created_at_utc:
        ISO-8601 UTC timestamp set at object creation time.
    """
    experiment_id: str
    sections: Dict[str, SectionResult] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

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

    Creates ``cfg.results_dir`` if it does not exist.  Existing files for
    the same experiment ID are overwritten, which allows re-running an
    experiment to update its results in place.

    Parameters
    ----------
    cfg:
        Experiment configuration (provides the output directory path).
    results:
        The completed experiment results to persist.

    Returns
    -------
    Path
        The experiment results directory (``cfg.results_dir``).
    """
    out_dir = cfg.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "experiment_id": results.experiment_id,
        "description": cfg.description,
        "hypothesis": cfg.hypothesis,
        "created_at_utc": results.created_at_utc,
        "sections_run": list(results.sections.keys()),
        "best": best,
        "saved_files": saved_files,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    _update_manifest(cfg, metadata)
    return out_dir


def _update_manifest(cfg: Config, metadata: dict) -> None:
    """Append or update this run's record in the shared experiment manifest.

    The manifest is a single JSON file at the experiments root that accumulates
    one record per unique experiment ID.  It is read by the notebook's
    comparison cell (``experiment_manifest.json``) so that all runs are visible
    without having to scan subdirectories.

    If a previous run for the same experiment_id exists in the manifest it is
    removed first, so the manifest always shows the most recent result for each
    ID and does not accumulate duplicate entries.
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
    # Remove any existing record for this experiment so the new one takes its
    # place at position 0 (most recent first).
    manifest["runs"] = [
        r for r in manifest["runs"] if r.get("experiment_id") != metadata["experiment_id"]
    ]
    manifest["runs"].insert(0, metadata)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
