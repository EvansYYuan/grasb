#!/usr/bin/env python
"""Assemble the reviewed Figure 2 panels into one publication-scale figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import pandas as pd

import plot_drift_expression_context_v1 as style
import plot_figure2_ab_v1 as ab
import plot_figure2_cd_expression_umaps_v1 as cd


REPO = Path(__file__).resolve().parents[3]
OUTPUT_STEM = REPO / "manuscript/figures/figure2_combined_v1"


def add_shared_key(fig: plt.Figure) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="#CDD2D4", markeredgewidth=0, label="other cells"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="#368F86", markeredgewidth=0,
               label="selected cell population"),
    ]
    fig.legend(handles=handles, loc="lower left", ncol=2, frameon=False,
               bbox_to_anchor=(0.18, 0.012), fontsize=6.0,
               columnspacing=1.1, handletextpad=0.35)
    cax = fig.add_axes([0.64, 0.028, 0.24, 0.008])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=cd.EXPRESSION_NORM, cmap=cd.EXPRESSION_CMAP),
        cax=cax, orientation="horizontal",
    )
    colorbar.set_ticks([0, 0.5, 1], labels=["low", "", "high"])
    colorbar.ax.tick_params(labelsize=5.3, length=0, pad=1.2)
    colorbar.outline.set_linewidth(0.35)
    colorbar.set_label("expression level of selected gene", fontsize=5.8, labelpad=1.8)


def main() -> None:
    style.register_fonts()
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(ab.INPUT)
    p_coords, p_expr = cd.load_pancreas()
    z_coords, z_expr = cd.load_zebrafish()

    # A tall single-column composition preserves readable rank labels above and
    # gives the denser expression panels more vertical area below. One shared
    # key replaces duplicated legends in c and d.
    fig = plt.figure(figsize=(style.WIDTH_IN, 9.90))
    outer = GridSpec(
        5, 1, figure=fig, left=0.07, right=0.985, top=0.975, bottom=0.085,
        # The first spacer is deliberately larger because the rank panels'
        # x-axis labels extend below their axes and otherwise consume the gap.
        height_ratios=[0.92, 0.28, 1.02, 0.08, 1.08], hspace=0.08,
    )

    rank_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.34)
    ax_a = fig.add_subplot(rank_grid[0, 0])
    ax_b = fig.add_subplot(rank_grid[0, 1])
    ab.plot_panel(ax_a, selection, "pancreas", "a")
    ab.plot_panel(ax_b, selection, "zebrafish", "b")

    cd.plot_species(fig, outer[2], p_coords, p_expr, cd.PANCREAS_SPECS, (2, 3),
                    "c", "Pancreas", "time", "day", point_size=0.72)
    cd.plot_species(fig, outer[4], z_coords, z_expr, cd.ZEBRAFISH_SPECS, (2, 4),
                    "d", "Zebrafish", "HPF", "hpf", point_size=1.05)
    add_shared_key(fig)
    style.save(fig, OUTPUT_STEM)


if __name__ == "__main__":
    main()
