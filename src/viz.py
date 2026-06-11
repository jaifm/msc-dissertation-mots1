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
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config
from .curve_metrics import _eval_curve
from .experiments.results import SectionResult
from .physics.models import fit_constrained_cy


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

    # Prefer test_mse when available; fall back to test_mae (physics_augmented section).
    error_col = "test_mse" if "test_mse" in ms.columns else "test_mae"
    error_label = "MSE" if error_col == "test_mse" else "MAE"
    for ax, metric, label in zip(
        axes, ["test_r2", error_col, "test_rmse"], ["R²", error_label, "RMSE"]
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


def plot_predicted_vs_actual(section: SectionResult, out_dir: Path) -> list:
    """Scatter plot of predicted vs actual values, one figure per physics model.

    Each figure contains one subplot per target for that model.  An identity
    line (y = x, dashed grey) marks perfect prediction.  For GPR sections
    whose predictions include ``y_lower_95`` / ``y_upper_95`` columns,
    asymmetric 95% prediction intervals are drawn as vertical error bars.

    Parameters
    ----------
    section:
        SectionResult with a non-empty ``predictions`` DataFrame.
    out_dir:
        Directory in which to save the PNGs.

    Returns
    -------
    list of Path
        One path per physics model figure saved.  Empty list if predictions
        is None or empty.
    """
    out_dir = _ensure_dir(out_dir)
    if section.predictions is None or section.predictions.empty:
        return []

    preds = section.predictions
    has_intervals = {"y_lower_95", "y_upper_95"}.issubset(preds.columns)
    paths = []

    for model_key, model_preds in preds.groupby("model_key"):
        targets = model_preds["target"].unique()
        n = len(targets)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
        model_name = model_preds["model_name"].iloc[0]
        fig.suptitle(f"{model_name} — Predicted vs Actual", fontweight="bold", fontsize=13)

        for ax_idx, target in enumerate(targets):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            sub = model_preds[model_preds["target"] == target]
            y_true = sub["y_true"].to_numpy()
            y_pred = sub["y_pred"].to_numpy()

            if has_intervals:
                lower_err = np.maximum(y_pred - sub["y_lower_95"].to_numpy(), 0)
                upper_err = np.maximum(sub["y_upper_95"].to_numpy() - y_pred, 0)
                ax.errorbar(
                    y_true, y_pred,
                    yerr=[lower_err, upper_err],
                    fmt="o", color="steelblue", ecolor="lightsteelblue",
                    elinewidth=1.2, capsize=3, markersize=5, alpha=0.85,
                    label="pred ± 95% PI",
                )
            else:
                ax.scatter(y_true, y_pred, color="steelblue", s=35, alpha=0.8)

            # Identity line spanning the combined range of true and predicted values.
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    "--", color="grey", linewidth=1, label="y = x")
            ax.set_xlabel("Actual", fontsize=9)
            ax.set_ylabel("Predicted", fontsize=9)
            ax.set_title(target, fontsize=10, fontweight="bold")
            if has_intervals:
                ax.legend(fontsize=7, loc="upper left")

        # Hide any unused subplot axes in the last row.
        for ax_idx in range(len(targets), nrows * ncols):
            axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

        plt.tight_layout()
        path = out_dir / f"{section.name}_{model_key}_predicted_vs_actual.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def plot_gpr_coverage(section: SectionResult, out_dir: Path) -> Optional[Path]:
    """Calibration chart: actual coverage vs nominal level for each target.

    Three groups of bars per target show what fraction of test points fell
    inside the 68%, 90%, and 95% prediction intervals.  Horizontal dashed
    reference lines mark the nominal levels.  A well-calibrated GPR should
    hit the reference lines; under-coverage (bars below the line) means the
    GPR is overconfident; over-coverage (bars above) means it is conservative.

    Returns None if the per_target DataFrame does not contain the coverage
    columns (e.g. for XGBoost-only sections).

    Parameters
    ----------
    section:
        SectionResult whose ``per_target`` contains coverage_68, coverage_90,
        coverage_95 columns (added by ``fit_and_evaluate_gpr``).
    out_dir:
        Directory in which to save the PNG.

    Returns
    -------
    Path or None
    """
    out_dir = _ensure_dir(out_dir)
    required = {"coverage_68", "coverage_90", "coverage_95"}
    if not required.issubset(section.per_target.columns):
        return None

    pt = section.per_target.copy()
    # One row per (model, target); average across models for the summary view.
    summary = (
        pt.groupby("target")[["coverage_68", "coverage_90", "coverage_95"]]
        .mean()
        .reset_index()
    )
    targets = summary["target"].tolist()
    n = len(targets)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    colors = {"68%": "#5B9BD5", "90%": "#ED7D31", "95%": "#70AD47"}
    nominals = {"68%": 0.68, "90%": 0.90, "95%": 0.95}

    for i, (label, col) in enumerate(
        [("68%", "coverage_68"), ("90%", "coverage_90"), ("95%", "coverage_95")]
    ):
        bars = ax.bar(
            x + (i - 1) * width,
            summary[col],
            width,
            label=f"{label} PI",
            color=colors[label],
            alpha=0.85,
        )
        # Annotate each bar with its numeric coverage value.
        for bar, val in zip(bars, summary[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    # Reference lines at nominal levels.
    for label, nom in nominals.items():
        ax.axhline(nom, linestyle="--", linewidth=1.2,
                   color=colors[label], alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Coverage (fraction of test points inside PI)", fontweight="bold")
    ax.set_title("GPR Prediction Interval Calibration", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(title="Interval", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = out_dir / f"{section.name}_gpr_coverage_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rheology_curves(
    predictions: pd.DataFrame,
    physics_test_by_model: dict,
    cfg,
    out_dir: Path,
) -> list:
    """Plot GPR-predicted viscosity curves vs actual measurements.

    One figure per physics model, faceted by Solid_Content_pct (one subplot
    per solid content level).  Within each subplot, curves are coloured by
    dispersant type so individual formulations are distinguishable.

      - Solid line   : viscosity curve from GPR-predicted parameters.
      - Dashed line  : viscosity curve from the fitted (true) physics
                       parameters — what the GPR was trained to predict.
      - Scatter dots : actual measured viscosity at 1, 10, 100 1/s.

    The gap between solid and dashed lines is the GPR prediction error; the
    gap between dashed lines and dots is the physics model fit residual.

    Parameters
    ----------
    predictions:
        ``SectionResult.predictions`` from a GPR or ensemble section.
        If ``row_idx`` is absent (pre-change run), it is inferred from
        within-group rank.
    physics_test_by_model:
        Dict mapping model_key → test-split physics DataFrame.
    cfg:
        Experiment configuration (viscosity column names, shear rates).
    out_dir:
        Directory in which to save the PNGs.

    Returns
    -------
    list of Path
        One path per physics model plotted.
    """
    from matplotlib.lines import Line2D

    out_dir = _ensure_dir(out_dir)

    # --- Inline physics model equations (mirrors src/physics/models.py) ------
    def _power_law(g, k, n):
        return k * g ** (n - 1)

    def _sisko(g, eta_inf, k, n):
        return eta_inf + k * g ** (n - 1)

    def _williamson(g, eta_0, eta_inf, lam, n):
        return eta_inf + (eta_0 - eta_inf) / (1 + (lam * g) ** n)

    _model_info = {
        "power_law":  ("Power Law",  _power_law,  ["k_consistency", "n_flow"]),
        "sisko":      ("Sisko",      _sisko,      ["eta_inf_sisko", "k_sisko", "n_sisko"]),
        "williamson": ("Williamson", _williamson, ["eta_0", "eta_inf", "lambda_time", "n_w"]),
    }

    # Distinct colours for each dispersant type.
    _dispersant_colors = {
        "Hypermer": "#E64B35",
        "Tego":     "#4DBBD5",
        "Triton":   "#00A087",
        "none":     "#7E6148",
    }
    _fallback_colors = sns.color_palette("husl", 8)

    visc_cols   = cfg.data.viscosity_columns
    shear_data  = np.array(cfg.physics.viscosity_shears)
    gamma_range = np.logspace(-1, 3, 300)

    paths = []
    for model_key, (display_name, curve_fn, param_names) in _model_info.items():
        model_preds = predictions[predictions["model_key"] == model_key]
        if model_preds.empty or model_key not in physics_test_by_model:
            continue

        physics_test = physics_test_by_model[model_key].reset_index(drop=True)

        if "row_idx" not in model_preds.columns:
            model_preds = model_preds.copy()
            model_preds["row_idx"] = model_preds.groupby("target").cumcount()

        pred_pivot = model_preds.pivot_table(
            index="row_idx", columns="target", values="y_pred", aggfunc="first"
        )
        true_pivot = model_preds.pivot_table(
            index="row_idx", columns="target", values="y_true", aggfunc="first"
        )

        solid_levels = sorted(physics_test["Solid_Content_pct"].unique())
        n_panels = len(solid_levels)
        ncols = min(n_panels, 3)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 5 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"{display_name} — GPR Predicted vs Measured Viscosity",
            fontweight="bold", fontsize=13, y=1.01,
        )

        # Track which dispersant types appear so we build a shared legend.
        seen_dispersants: dict = {}

        for panel_idx, solid in enumerate(solid_levels):
            ax = axes[panel_idx // ncols][panel_idx % ncols]
            mask = physics_test["Solid_Content_pct"] == solid
            subset_idx = physics_test.index[mask].tolist()

            for row_idx in subset_idx:
                if row_idx not in pred_pivot.index:
                    continue
                row = physics_test.iloc[row_idx]
                disp = str(row.get("Dispersent_Type", "none"))
                colour = _dispersant_colors.get(
                    disp, _fallback_colors[hash(disp) % len(_fallback_colors)]
                )
                seen_dispersants[disp] = colour

                visc_meas = [row[c] for c in visc_cols if c in row.index]
                ax.scatter(
                    shear_data[: len(visc_meas)], visc_meas,
                    color=colour, s=60, zorder=5,
                    edgecolors="k", linewidths=0.6,
                )

                if all(p in true_pivot.columns for p in param_names):
                    p_true = [float(true_pivot.loc[row_idx, p]) for p in param_names]
                    try:
                        ax.plot(
                            gamma_range, curve_fn(gamma_range, *p_true),
                            "--", color=colour, alpha=0.55, linewidth=1.2,
                        )
                    except Exception:
                        pass

                if all(p in pred_pivot.columns for p in param_names):
                    p_pred = [float(pred_pivot.loc[row_idx, p]) for p in param_names]
                    try:
                        ax.plot(
                            gamma_range, curve_fn(gamma_range, *p_pred),
                            "-", color=colour, alpha=0.85, linewidth=1.8,
                        )
                    except Exception:
                        pass

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.1, 1000)
            ax.set_xlabel("Shear Rate (1/s)", fontsize=9)
            ax.set_ylabel("Viscosity (Pa·s)", fontsize=9)
            ax.set_title(f"Solid Content = {solid:.0f}%", fontweight="bold", fontsize=10)
            ax.grid(True, which="both", alpha=0.2)

        # Hide unused subplots.
        for panel_idx in range(n_panels, nrows * ncols):
            axes[panel_idx // ncols][panel_idx % ncols].set_visible(False)

        # Shared legend: dispersant type + line style guide.
        legend_handles = [
            Line2D([0], [0], color=col, lw=2, label=disp)
            for disp, col in sorted(seen_dispersants.items())
        ] + [
            Line2D([0], [0], color="grey", lw=1.8, ls="-",  label="GPR predicted"),
            Line2D([0], [0], color="grey", lw=1.2, ls="--", label="Physics fit (actual)"),
            Line2D([0], [0], marker="o", color="grey", ls="None",
                   ms=7, markeredgecolor="k", label="Measured"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.04),
        )

        plt.tight_layout()
        path = out_dir / f"rheology_curves_{model_key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def plot_curve_rmse_comparison(section: SectionResult, out_dir: Path) -> Optional[Path]:
    """Horizontal grouped bar chart comparing curve-space RMSE across models.

    Two bars per model:
      - ``mean_curve_rmse_cross``  (blue)  — deployment value vs Constrained CY
      - ``mean_curve_rmse_within`` (orange) — ML quality vs within-model scipy fit

    Models are sorted by ascending cross-model RMSE (best deployment value at
    the top).  Lower is better for both metrics.

    Only rendered when the section has a non-empty ``curve_summary`` DataFrame
    containing both columns.  Returns None silently otherwise.
    """
    if section.curve_summary is None or section.curve_summary.empty:
        return None
    cs = section.curve_summary
    if "mean_curve_rmse_cross" not in cs.columns or "mean_curve_rmse_within" not in cs.columns:
        return None

    cs = cs.sort_values("mean_curve_rmse_cross", ascending=True).reset_index(drop=True)
    models = cs["model_name"].tolist()
    n = len(models)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.7 * n + 1.5)))
    y = np.arange(n)
    bar_h = 0.35

    bars_cross = ax.barh(
        y + bar_h / 2, cs["mean_curve_rmse_cross"], bar_h,
        label="Cross-model (vs Constrained CY)", color="#4C72B0",
    )
    bars_within = ax.barh(
        y - bar_h / 2, cs["mean_curve_rmse_within"], bar_h,
        label="Within-model (vs scipy fit)", color="#DD8452",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Mean log-RMSE  (log Pa·s)")
    ax.set_title(f"Curve-Space RMSE — {section.name}")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    # Value labels on bars
    for bar in bars_cross:
        w = bar.get_width()
        if not np.isnan(w):
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left", fontsize=8)
    for bar in bars_within:
        w = bar.get_width()
        if not np.isnan(w):
            ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{w:.3f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    path = out_dir / f"{section.name}_curve_rmse_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predicted_curves(
    section: SectionResult,
    physics_datasets: Dict[str, Dict[str, pd.DataFrame]],
    cfg: Config,
    out_dir: Path,
) -> list:
    """Overlay predicted viscosity curves against reference curves for all test rows.

    One PNG per physics model.  Each figure overlays three curve families across
    every test row, giving a spaghetti view of how well the ML-predicted curves
    track the reference:

      ML predicted (blue, solid)
          Curves from the ML-predicted parameter vectors.

      Within-model scipy (orange, dashed)
          Curves from the scipy-fitted parameter vectors already stored in the
          physics dataset (y_true).  Shows whether the ML recovered the model's
          own parameters accurately.

      Cross-model reference (grey, dotted)
          Constrained CY (η_∞=0.01, a=2) fitted to the three raw viscosity
          measurements.  The common deployment benchmark.

    Raw measurement points (black dots) — 3 per row — are overlaid on every plot.
    Bold lines show the pointwise median of each family across the test set.

    Parameters
    ----------
    section:
        SectionResult whose ``predictions`` DataFrame supplies the ML-predicted
        parameters.  Returns an empty list if ``predictions`` is None.
    physics_datasets:
        Nested dict ``{model_key: {'train': df, 'test': df}}`` produced by
        ``build_all_physics_datasets``.  Provides raw viscosity columns and
        scipy-fitted parameters for the test set.
    cfg:
        Experiment configuration.  Provides ``cfg.physics.targets``,
        ``cfg.data.viscosity_columns``, and ``cfg.physics.viscosity_shears``.
    out_dir:
        Directory in which to save the PNGs.

    Returns
    -------
    list of Path
        One path per model for which curves could be plotted.
    """
    if section.predictions is None or section.predictions.empty:
        return []

    shear_grid = np.logspace(np.log10(0.1), np.log10(1000.0), 200)
    visc_cols = cfg.data.viscosity_columns
    shear_rates = np.array(cfg.physics.viscosity_shears)

    preds_all = section.predictions
    if "model_key" not in preds_all.columns:
        return []

    paths = []
    for model_key in preds_all["model_key"].unique():
        if model_key not in physics_datasets:
            continue

        target_cols = cfg.targets_for(model_key)
        test_physics = physics_datasets[model_key]["test"]
        display_name = cfg.display_name(model_key)

        model_preds = preds_all[preds_all["model_key"] == model_key]
        try:
            preds_wide = model_preds.pivot(
                index="row_idx", columns="target", values="y_pred"
            )
        except Exception:
            continue

        true_wide = test_physics[target_cols].reset_index(drop=True)
        raw_visc = test_physics[visc_cols].reset_index(drop=True)

        ml_curves, within_curves, cross_curves = [], [], []
        meas_gamma, meas_eta = [], []

        for pos in range(len(preds_wide)):
            row_key = preds_wide.index[pos]
            try:
                pred_params = preds_wide.loc[row_key, target_cols].to_numpy(dtype=float)
                true_params = true_wide.iloc[pos].to_numpy(dtype=float)
                visc_vals = raw_visc.iloc[pos].to_numpy(dtype=float)

                ml_eta = _eval_curve(model_key, pred_params, shear_grid)
                within_eta = _eval_curve(model_key, true_params, shear_grid)
                ref_params = fit_constrained_cy(visc_vals, shear_rates)
                cross_eta = _eval_curve("constrained_cy", ref_params, shear_grid)

                if np.all(ml_eta > 0):
                    ml_curves.append(ml_eta)
                if np.all(within_eta > 0):
                    within_curves.append(within_eta)
                if np.all(cross_eta > 0):
                    cross_curves.append(cross_eta)

                for g, v in zip(shear_rates, visc_vals):
                    meas_gamma.append(g)
                    meas_eta.append(v)
            except Exception:
                continue

        if not ml_curves:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))

        # Individual curves (low alpha for readability)
        for eta in ml_curves:
            ax.plot(shear_grid, eta, color="#4C72B0", alpha=0.12, lw=0.8)
        for eta in within_curves:
            ax.plot(shear_grid, eta, color="#DD8452", alpha=0.12, lw=0.8, ls="--")
        for eta in cross_curves:
            ax.plot(shear_grid, eta, color="#888888", alpha=0.10, lw=0.8, ls=":")

        # Median lines (bold)
        ax.plot(
            shear_grid, np.median(ml_curves, axis=0),
            color="#4C72B0", lw=2.0, label=f"ML predicted  (n={len(ml_curves)}, median)",
        )
        if within_curves:
            ax.plot(
                shear_grid, np.median(within_curves, axis=0),
                color="#DD8452", lw=2.0, ls="--",
                label=f"Within-model scipy  (n={len(within_curves)}, median)",
            )
        if cross_curves:
            ax.plot(
                shear_grid, np.median(cross_curves, axis=0),
                color="#555555", lw=2.0, ls=":",
                label=f"Constrained CY ref  (n={len(cross_curves)}, median)",
            )

        # Raw measurement dots
        if meas_gamma:
            ax.scatter(
                meas_gamma, meas_eta,
                color="black", s=14, zorder=5, alpha=0.45,
                label="Raw measurements",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Shear rate  (1/s)")
        ax.set_ylabel("Viscosity  (Pa·s)")
        ax.set_title(f"{display_name} — predicted vs reference curves  [{section.name}]")
        ax.legend(fontsize=8, loc="lower left")
        sns.despine(ax=ax)

        plt.tight_layout()
        path = out_dir / f"{section.name}_{model_key}_curve_overlay.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return paths


def plot_augmented_dataset_overview(
    splits: Dict,
    out_dir: Path,
) -> Path:
    """Scatter of all training points (measured + augmented) in log shear-rate space.

    One subplot per physics oracle.  Measured points (3 per row) are drawn as
    large filled circles; augmented points as small semi-transparent dots.  Both
    axes are log10-transformed so the density of the augmented grid is visible.

    Parameters
    ----------
    splits : dict
        Mapping ``{model_key: AugmentedSplit}`` as returned by
        ``data.build_augmented_split``.
    out_dir : Path
        Directory to save the PNG.

    Returns
    -------
    Path to saved PNG.
    """
    out_dir = _ensure_dir(out_dir)
    oracles = list(splits.keys())
    n = len(oracles)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    fig.suptitle("Physics-augmented training set — log γ̇ vs log η", fontweight="bold")

    for col, oracle_key in enumerate(oracles):
        split = splits[oracle_key]
        ax = axes[0][col]
        df = split.train_long.copy()
        measured = df[df["source"] == "measured"]
        augmented = df[df["source"] != "measured"]

        ax.scatter(
            augmented["shear_rate"], augmented["viscosity"],
            s=6, alpha=0.25, color="steelblue", label=f"augmented (n={len(augmented):,})",
        )
        ax.scatter(
            measured["shear_rate"], measured["viscosity"],
            s=40, alpha=0.9, color="crimson", zorder=3,
            label=f"measured (n={len(measured):,})",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Shear rate γ̇ (s⁻¹)", fontsize=10)
        ax.set_ylabel("Viscosity η (Pa·s)", fontsize=10)
        ax.set_title(oracle_key.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    path = out_dir / "augmented_dataset_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_augmented_sample_curves(
    splits: Dict,
    out_dir: Path,
    *,
    n_sample: int = 12,
    seed: int = 0,
) -> Path:
    """Flow curves for a random sample of training rows.

    For each sampled row one subplot is drawn showing the viscosity–shear-rate
    relationship on log-log axes.  Each physics oracle contributes:

    * A **line** connecting the 30 augmented points (sorted by shear rate) —
      this traces the oracle's fitted curve through the shear range.
    * **Filled circles** for the three measured viscosities.

    Parameters
    ----------
    splits : dict
        Mapping ``{model_key: AugmentedSplit}``.
    out_dir : Path
    n_sample : int
        Number of training rows to display.
    seed : int
        Reproducibility seed for the random row sample.

    Returns
    -------
    Path to saved PNG.
    """
    out_dir = _ensure_dir(out_dir)

    # Use source_row_idx (unique integer per training row) when available;
    # fall back to row_id for backwards compatibility with older augmented frames.
    first_split = next(iter(splits.values()))
    use_src_idx = "source_row_idx" in first_split.train_long.columns
    if use_src_idx:
        all_keys = sorted(first_split.train_long["source_row_idx"].unique())
    else:
        all_keys = sorted(first_split.train_long["row_id"].unique())

    rng = np.random.default_rng(seed)
    sampled_keys = list(rng.choice(all_keys,
                                   size=min(n_sample, len(all_keys)),
                                   replace=False))

    ncols = min(4, n_sample)
    nrows = int(np.ceil(len(sampled_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle("Physics-augmented flow curves — sample training rows",
                 fontweight="bold", fontsize=13)

    _ORACLE_COLORS = {
        "power_law": "#1f77b4",
        "constrained_cy": "#ff7f0e",
    }
    _DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, key in enumerate(sampled_keys):
        ax = axes[idx // ncols][idx % ncols]
        row_id_for_title = None
        for o_idx, (oracle_key, split) in enumerate(splits.items()):
            color = _ORACLE_COLORS.get(oracle_key,
                                       _DEFAULT_COLORS[o_idx % len(_DEFAULT_COLORS)])
            if use_src_idx:
                sub = split.train_long[split.train_long["source_row_idx"] == key]
            else:
                sub = split.train_long[split.train_long["row_id"] == key]
            if sub.empty:
                continue
            if row_id_for_title is None:
                row_id_for_title = sub["row_id"].iloc[0]
            aug = sub[sub["source"] != "measured"].sort_values("shear_rate")
            meas = sub[sub["source"] == "measured"].sort_values("shear_rate")
            label = oracle_key.replace("_", " ").title()
            if not aug.empty:
                ax.plot(aug["shear_rate"], aug["viscosity"],
                        "-", color=color, linewidth=1.5,
                        label=label, alpha=0.85)
            if not meas.empty:
                ax.scatter(meas["shear_rate"], meas["viscosity"],
                           s=55, color=color, zorder=4, edgecolors="white",
                           linewidths=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("γ̇ (s⁻¹)", fontsize=8)
        ax.set_ylabel("η (Pa·s)", fontsize=8)
        short_id = str(row_id_for_title or key).replace("_", " ")
        if len(short_id) > 22:
            short_id = short_id[:20] + "…"
        ax.set_title(short_id, fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.4, alpha=0.55)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes.
    for idx in range(len(sampled_keys), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    path = out_dir / "augmented_sample_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_all_plots(
    results,
    out_dir: Path,
    *,
    physics_datasets: Optional[Dict] = None,
    cfg: Optional[Config] = None,
) -> list:
    """Run every applicable plot function for every section in ``results``.

    Skips optional charts (heatmap, variants, curve overlay) when they return
    None or when required inputs are not provided.  All generated files are
    saved to ``out_dir`` and their paths are returned so the caller can display
    or log them.

    Parameters
    ----------
    results:
        An ``ExperimentResults`` object containing one or more sections.
    out_dir:
        Root directory in which to save all PNGs (typically ``cfg.results_dir``).
    physics_datasets:
        Optional.  When provided together with ``cfg``, the predicted-curve
        overlay plots (``plot_predicted_curves``) are also generated.
    cfg:
        Experiment configuration.  Required for ``plot_predicted_curves``.

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
        paths.extend(plot_predicted_vs_actual(section, out_dir))
        c = plot_gpr_coverage(section, out_dir)
        if c:
            paths.append(c)
        cr = plot_curve_rmse_comparison(section, out_dir)
        if cr:
            paths.append(cr)
        if physics_datasets is not None and cfg is not None:
            paths.extend(plot_predicted_curves(section, physics_datasets, cfg, out_dir))
    return paths
