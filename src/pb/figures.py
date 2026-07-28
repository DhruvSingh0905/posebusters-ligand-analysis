"""Figures for the report.

Colour discipline: the two method classes are a categorical pair in fixed order
(a filter that drops one must not repaint the other); check failure rates across
an ordered ligand partition are magnitude, so they get a single hue; Cohen's d is
signed, so it gets a diverging map with two hues and a neutral midpoint.
"""

from __future__ import annotations

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import paths
from .analyze import check_descriptor_associations, outcome_by_bin, primary
from .build import CHECKS

log = logging.getLogger(__name__)

INK = "#12202A"
MUTED = "#6C7E87"
GRID = "#D9E1E3"
ACCENT = "#0A6B87"
CLASSES = {"classical": "#0A6B87", "deep learning": "#B4502F"}
DIVERGING = LinearSegmentedColormap.from_list(
    "teal_oxide", ["#0A6B87", "#7FA3B0", "#E8E9E6", "#C79A7C", "#8E3116"]
)

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.linewidth": 0.8,
    "axes.titlesize": 9.5,
    "axes.titleweight": "semibold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.frameon": False,
})

BIN_LABEL = {
    "rotb_bin": "rotatable bonds",
    "mw_bin": "molecular weight (Da)",
    "rings_bin": "ring count",
    "stereo_bin": "stereocentres",
    "heavy_bin": "heavy atoms",
}


def _strip(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(False)


def fig_gap_by_partition(raw: pd.DataFrame) -> None:
    """The headline: accurate-but-invalid rate across every ligand partition."""
    partitions = ["rotb_bin", "mw_bin", "rings_bin", "stereo_bin"]
    fig, axes = plt.subplots(1, len(partitions), figsize=(11, 2.9), sharey=True)

    for ax, column in zip(axes, partitions):
        table = outcome_by_bin(raw, column)
        for method_class, colour in CLASSES.items():
            sub = table[table.method_class == method_class]
            levels = [str(v) for v in sub[column]]
            ax.plot(levels, sub["invalid_given_accurate"], marker="o",
                    markersize=5, linewidth=2, color=colour, label=method_class,
                    markeredgecolor="white", markeredgewidth=1.2)
            for x, y in zip(levels, sub["invalid_given_accurate"]):
                if pd.notna(y):
                    ax.annotate(
                        f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, 9 if method_class == "deep learning" else -14),
                        ha="center", fontsize=7, color=colour, weight="bold",
                    )
        ax.set_xlabel(BIN_LABEL[column], labelpad=6)
        ax.set_ylim(-0.14, 1.16)
        _strip(ax)

    axes[0].set_ylabel("of accurate poses,\nshare physically invalid")
    axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.11))
    fig.suptitle(
        "Passing RMSD ≤ 2 Å means less and less as the ligand gets harder",
        x=0.5, y=1.05, fontsize=11, weight="semibold",
    )
    fig.savefig(paths.FIGURES / "01_gap_by_partition.png")
    plt.close(fig)


def fig_checks_by_flexibility(dl: pd.DataFrame) -> None:
    """Which specific checks degrade with rotatable bonds."""
    order = [c for c in CHECKS if (dl[c] == False).sum() >= 60]  # noqa: E712
    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.5 * nrows), sharey=True)
    flat = axes.ravel()

    for ax, check in zip(flat, order):
        ran = dl[dl[check].notna()]
        rate = (
            ran.assign(failed=(ran[check] == False))  # noqa: E712
            .groupby("rotb_bin", observed=True)["failed"]
            .mean()
        )
        levels = [str(v) for v in rate.index]
        ax.bar(levels, rate.to_numpy(), color=ACCENT, width=0.66)
        for x, y in zip(levels, rate.to_numpy()):
            ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=6.5, color=INK)
        ax.set_title(check.replace("_", " "), fontsize=8)
        ax.set_ylim(0, 1.14)
        _strip(ax)

    for ax in flat[len(order):]:
        ax.set_visible(False)
    for ax in axes[-1] if nrows > 1 else axes:
        ax.set_xlabel("rotatable bonds")
    (axes[0][0] if nrows > 1 else axes[0]).set_ylabel("failure rate")
    (axes[0][0] if nrows > 1 else axes[0]).yaxis.set_major_formatter(
        mpl.ticker.PercentFormatter(1.0)
    )

    fig.suptitle(
        "Deep-learning failure modes, by ligand flexibility",
        x=0.5, y=1.0, fontsize=11, weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(paths.FIGURES / "02_checks_by_flexibility.png")
    plt.close(fig)


def fig_association_heatmap(dl: pd.DataFrame) -> None:
    """Cohen's d for every (check, descriptor) pair."""
    assoc = check_descriptor_associations(dl)
    grid = assoc.pivot(index="check", columns="descriptor", values="cohens_d")

    # order rows by how often the check fails, columns by strongest effect
    rates = assoc.groupby("check")["failure_rate"].first().sort_values(ascending=False)
    grid = grid.loc[rates.index]
    grid = grid[grid.abs().max().sort_values(ascending=False).index]

    limit = float(np.nanmax(np.abs(grid.to_numpy())))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(grid) + 2.2))
    mesh = ax.imshow(
        grid.to_numpy(), cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto",
    )

    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels([c.replace("_", " ") for c in grid.columns],
                       rotation=42, ha="right", fontsize=7)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(
        [f"{c.replace('_', ' ')}  ({rates[c]:.0%})" for c in grid.index], fontsize=7.5
    )
    ax.grid(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.to_numpy()[i, j]
            if np.isfinite(value) and abs(value) >= 0.5:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(value) > 0.75 * limit else INK)

    bar = fig.colorbar(mesh, ax=ax, fraction=0.018, pad=0.012)
    bar.set_label("Cohen's d   (positive → failing poses score higher)", fontsize=7.5)
    bar.outline.set_visible(False)

    ax.set_title(
        "What kind of molecule trips each check\n"
        "deep-learning poses, 428 complexes; row label shows how often the check fails",
        fontsize=10.5, weight="semibold", loc="left", pad=12,
    )
    fig.savefig(paths.FIGURES / "03_association_heatmap.png")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    paths.ensure_dirs()

    df = pd.read_parquet(paths.JOINED_PARQUET)
    raw = primary(df, post="none")
    dl = raw[raw["method_class"] == "deep learning"]

    fig_gap_by_partition(raw)
    fig_checks_by_flexibility(dl)
    fig_association_heatmap(dl)

    for path in sorted(paths.FIGURES.glob("*.png")):
        log.info("wrote %s (%.0f kB)", path.name, path.stat().st_size / 1000)


if __name__ == "__main__":
    main()
