#!/usr/bin/env python
"""Assemble Figure 2 v2 with the expanded two-sided rank annotations."""

from pathlib import Path

import plot_figure2_ab_v2 as ab
import plot_figure2_combined_v1 as base


REPO = Path(__file__).resolve().parents[3]

# Reuse the reviewed composition while replacing only the rank-panel renderer
# and output stem. This leaves the expression UMAP panels unchanged.
base.ab = ab
base.OUTPUT_STEM = REPO / "manuscript/figures/figure2_combined_v2"
_add_expression_key = base.add_shared_key


def add_all_keys(fig):
    _add_expression_key(fig)
    ab.add_rank_key(fig, (0.5, 0.670))


base.add_shared_key = add_all_keys


def main() -> None:
    base.style.register_fonts()
    base.OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)

    selection = base.pd.read_csv(ab.INPUT)
    p_coords, p_expr = base.cd.load_pancreas()
    z_coords, z_expr = base.cd.load_zebrafish()

    # Keep a meaningful legend row while compacting the full canvas and the
    # inter-row padding around it.
    fig = base.plt.figure(figsize=(base.style.WIDTH_IN, 9.80))
    outer = base.GridSpec(
        5, 1, figure=fig, left=0.07, right=0.985, top=0.975, bottom=0.082,
        height_ratios=[0.92, 0.34, 1.02, 0.08, 1.08], hspace=0.02,
    )

    rank_grid = base.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.34
    )
    ax_a = fig.add_subplot(rank_grid[0, 0])
    ax_b = fig.add_subplot(rank_grid[0, 1])
    ab.plot_panel(ax_a, selection, "pancreas", "a")
    ab.plot_panel(ax_b, selection, "zebrafish", "b")

    base.cd.plot_species(
        fig, outer[2], p_coords, p_expr, base.cd.PANCREAS_SPECS, (2, 3),
        "c", "Pancreas", "time", "day", point_size=0.72,
    )
    base.cd.plot_species(
        fig, outer[4], z_coords, z_expr, base.cd.ZEBRAFISH_SPECS, (2, 4),
        "d", "Zebrafish", "HPF", "hpf", point_size=1.05,
    )
    base.add_shared_key(fig)
    base.style.save(fig, base.OUTPUT_STEM)


if __name__ == "__main__":
    main()
