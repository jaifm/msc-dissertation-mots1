"""Standalone plotting functions for experiment result visualisation.

OVERVIEW
--------
Each function in this module takes a ``SectionResult`` (or a subset of one)
and an output directory, generates one chart, saves it as a PNG, and returns
the saved path.  The ``make_all_plots`` convenience function runs every
applicable chart for every section in an ``ExperimentResults``.

CHART CATALOGUE
---------------
plot_model_comparison
    Three horizontal bar charts side-by-side showing mean test R², MSE, and
    RMSE for each model.  Models are sorted by R² (best at the top) so it is
    immediately clear which physics model XGBoost predicts most accurately.

plot_per_target_heatmap
    A colour-coded heatmap of test R² with targets on the y-axis and models
    on the x-axis.  Reveals which individual physics parameters (e.g. η_0 vs
    n vs λ) are easy or hard to predict from composition alone.

plot_train_vs_test
    Grouped bar chart comparing train R² and test R² for each model.  A large
    gap between train and test signals overfitting.

plot_sweep_variants
    Grouped bar chart comparing variants (e.g. 'baseline' vs 'log_numeric')
    for each model.  Only rendered when a 'variant' column is present in the
    model_summary, i.e. for the sweep section.

DESIGN NOTES
------------
All functions follow the same pattern so adding a new chart is mechanical:
write a function that takes (section, out_dir), produces a figure, calls
``fig.savefig``, and returns the path.  The ``make_all_plots`` function can
then be extended to call it.

DPI is set to 150 for all outputs, which balances file size against legibility
when the PNGs are embedded in a PDF dissertation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .experiments.results import SectionResult


def _ensure_dir(p: Path) -> Path:
    """Create the directory (and any missing parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_model_comparison(section: SectionResult, out_dir: Path) -> Path:
    """Generate three side-by-side horizontal bar charts: R², MSE, RMSE.

    Models are displayed in descending R² order (best performing at the top)
    so the ranking is immediately visible without reading the axis labels.
    Distinct colours from the 'husl' palette are used to make models easy to
    distinguish when the chart is printed in greyscale or viewed by
    colour-blind readers.

    Parameters
    ----------
    section:
        A SectionResult whose ``model_summary`` contains test_r2, test_mse,
        and test_rmse columns.
    out_dir:
        Directory in which to save the PNG.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    out_dir = _ensure_dir(out_dir)
    ms = section.sorted_by_r2()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = sns.color_palette("husl", len(ms))

    for ax, metric, label in zip(
        axes, ["test_r2", "test_mse", "test_rmse"], ["R²", "MSE", "RMSE"]
    ):
        ax.barh(ms["model_name"], ms[metric], color=colors)
        ax.set_xlabel(f"Mean Test {label}", fontweight="bold")
        ax.set_title(f"Model Performance: {label}", fontweight="bold")

    plt.tight_layout()
    path = out_dir / f"{section.name}_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_target_heatmap(section: SectionResult, out_dir: Path) -> Optional[Path]:
    """Generate an R² heatmap with targets as rows and models as columns.

    This chart exposes the structure that the model comparison bar charts
    hide: a model that averages R²=0.8 might achieve R²=0.95 on η_0 and
    only R²=0.6 on n, which has different implications for formulation design
    than a model that achieves R²=0.8 uniformly across all targets.

    Returns None (without error) if ``per_target`` is empty, which can happen
    if a section ran with only one target.

    Parameters
    ----------
    section:
        SectionResult with a non-empty ``per_target`` DataFrame containing
        columns [target, model_name, test_r2].
    out_dir:
        Directory in which to save the PNG.

    Returns
    -------
    Path or None
        Path to the saved PNG, or None if the DataFrame was empty.
    """
    out_dir = _ensure_dir(out_dir)
    if section.per_target.empty:
        return None
    pivot = section.per_target.pivot_table(
        values="test_r2", index="target", columns="model_name", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    # centre=0.5 places the midpoint of the green-yellow-red colour scale at
    # R²=0.5, which is a natural "mediocre" threshold for regression quality.
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", center=0.5, ax=ax,
                cbar_kws={"label": "R²"})
    ax.set_title("Test R² by Target and Model", fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"{section.name}_per_target_r2_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_train_vs_test(section: SectionResult, out_dir: Path) -> Path:
    """Generate a grouped bar chart comparing train R² and test R² per model.

    A large gap between the two bars for a given model indicates overfitting:
    the model has memorised the training data but fails to generalise.  This
    is particularly relevant for XGBoost on small datasets, where high tree
    depth can lead to near-perfect training R² with poor test R².

    Parameters
    ----------
    section:
        SectionResult whose ``model_summary`` contains both train_r2 and
        test_r2 columns.
    out_dir:
        Directory in which to save the PNG.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    out_dir = _ensure_dir(out_dir)
    ms = section.sorted_by_r2()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ms))
    width = 0.35
    ax.bar(x - width / 2, ms["train_r2"], width, label="Train R²", color="steelblue")
    ax.bar(x + width / 2, ms["test_r2"], width, label="Test R²", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(ms["model_name"], rotation=45, ha="right")
    ax.set_ylabel("R²", fontweight="bold")
    ax.set_title("Train vs Test R²", fontweight="bold")
    # Y-axis fixed at [0, 1] so charts from different experiments are directly
    # comparable without rescaling.
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()
    path = out_dir / f"{section.name}_train_vs_test_r2.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sweep_variants(section: SectionResult, out_dir: Path) -> Optional[Path]:
    """Generate a grouped bar chart comparing sweep variants per model.

    Only rendered when the ``model_summary`` contains a 'variant' column,
    which is added by the sweep section (``run_sweep``) but not the baseline
    section.  If the column is absent the function returns None silently so
    ``make_all_plots`` does not need to know which sections produce variants.

    The chart makes it immediately visible whether log-transforming numeric
    features improved or hurt performance for each model, which informs the
    feature engineering decision for subsequent experiments.

    Parameters
    ----------
    section:
        SectionResult from the sweep section (expects a 'variant' column).
    out_dir:
        Directory in which to save the PNG.

    Returns
    -------
    Path or None
    """
    out_dir = _ensure_dir(out_dir)
    if "variant" not in section.model_summary.columns:
        return None
    pivot = section.model_summary.pivot_table(
        index="model_name", columns="variant", values="test_r2"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean Test R²", fontweight="bold")
    ax.set_title("Variants Compared", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"{section.name}_variants.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_plots(results, out_dir: Path) -> list:
    """Run every applicable plot function for every section in ``results``.

    Skips optional charts (heatmap, variants) when they return None.  All
    generated files are saved to ``out_dir`` and their paths are returned so
    the caller can display or log them.

    Parameters
    ----------
    results:
        An ``ExperimentResults`` object containing one or more sections.
    out_dir:
        Root directory in which to save all PNGs (typically ``cfg.results_dir``).

    Returns
    -------
    list of Path
        Paths of all successfully saved chart files.
    """
    paths = []
    for section in results.sections.values():
        paths.append(plot_model_comparison(section, out_dir))
        h = plot_per_target_heatmap(section, out_dir)
        if h:
            paths.append(h)
        if {"train_r2", "test_r2"}.issubset(section.model_summary.columns):
            paths.append(plot_train_vs_test(section, out_dir))
        v = plot_sweep_variants(section, out_dir)
        if v:
            paths.append(v)
    return paths
