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
from .analyze import outcome_by_bin, primary
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


def fig_effect_forest(grid: pd.DataFrame, top_n: int = 18) -> None:
    """Surviving effects as points with cluster-bootstrap intervals, faceted by method."""
    surviving = grid[grid["fdr_reject"]].nlargest(top_n, "abs_d")
    if surviving.empty:
        log.warning("no effects survived FDR - skipping forest plot")
        return

    surviving = surviving.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{r.check.replace('_', ' ')} × {r.descriptor.replace('_', ' ')}  [{r.method}]"
        for r in surviving.itertuples()
    ]
    y = np.arange(len(surviving))

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(surviving) + 1.6))
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.hlines(y, surviving["lo"], surviving["hi"], color=ACCENT, linewidth=2, zorder=2)
    ax.plot(surviving["d"], y, "o", markersize=6, color=ACCENT,
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Cohen's d  (cluster-bootstrap 95% CI, BH-FDR controlled)")
    ax.yaxis.grid(False)
    _strip(ax)
    ax.set_title("Effects that survive eligibility gating, clustering and FDR",
                 fontsize=10.5, weight="semibold", loc="left", pad=10)
    fig.savefig(paths.FIGURES / "03_effect_forest.png")
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

    grid_path = paths.REPORTS / "tables" / "association_grid.csv"
    if grid_path.exists():
        grid = pd.read_csv(grid_path)
        fig_effect_forest(grid)
    else:
        log.warning("no association_grid.csv - run `python -m pb.analyze` first")

    for path in sorted(paths.FIGURES.glob("*.png")):
        log.info("wrote %s (%.0f kB)", path.name, path.stat().st_size / 1000)


if __name__ == "__main__":
    main()
