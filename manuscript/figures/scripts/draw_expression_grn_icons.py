#!/usr/bin/env python
"""Draw paper-ready gene-expression matrix and GRN schematic assets.

The artwork is constructed entirely from Matplotlib vector primitives, so it
has no external asset license or attribution requirement.

Run:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/draw_expression_grn_icons.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"

# Figure 1 visual grammar: muted biological colors, coral GRN emphasis,
# neutral graphite structure, and a warm transparent canvas.
TEAL = "#3FA89B"
BLUE = "#277DB7"
GOLD = "#D99F2B"
VIOLET = "#7C5AB5"
CORAL = "#F06455"
INK = "#60676B"
MID = "#92999C"
PALE = "#E7E9E8"
WHITE = "#FFFFFF"

HEATMAP = LinearSegmentedColormap.from_list(
    "expression", ["#F4F1EE", "#C9DCDA", "#72B1AA", "#2F8583"]
)


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def blank_ax(figsize: tuple[float, float]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def expression_matrix(ax: plt.Axes, *, label: bool = False) -> None:
    """Three stacked cells-by-genes matrices, echoing snapshot timepoints."""
    values = np.array(
        [
            [0.78, 0.91, 0.18, 0.61, 0.83, 0.24],
            [0.86, 0.54, 0.13, 0.72, 0.88, 0.31],
            [0.68, 0.82, 0.20, 0.79, 0.55, 0.27],
            [0.34, 0.71, 0.16, 0.63, 0.41, 0.19],
            [0.73, 0.48, 0.12, 0.86, 0.67, 0.22],
        ]
    )
    row_colors = [TEAL, BLUE, BLUE, GOLD, VIOLET]

    def draw_card(x0: float, y0: float, width: float, height: float,
                  *, alpha: float, zorder: int, phase: int) -> None:
        card = FancyBboxPatch(
            (x0, y0), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.036",
            facecolor=WHITE, edgecolor=MID, linewidth=1.45,
            alpha=alpha, zorder=zorder,
        )
        ax.add_patch(card)

        nrow, ncol = values.shape
        gx, gy = x0 + 0.052 * width, y0 + 0.095 * height
        gw, gh = width * 0.89, height * 0.82
        gap = width * 0.012
        cw = (gw - gap * (ncol - 1)) / ncol
        ch = (gh - gap * (nrow - 1)) / nrow
        shifted = np.roll(values, phase, axis=1)
        for r in range(nrow):
            for c in range(ncol):
                yy = gy + (nrow - 1 - r) * (ch + gap)
                ax.add_patch(
                    Rectangle(
                        (gx + c * (cw + gap), yy), cw, ch,
                        facecolor=HEATMAP(shifted[r, c]), edgecolor=WHITE,
                        linewidth=0.42, alpha=alpha, zorder=zorder + 0.2,
                    )
                )

        dot_x = x0 - width * 0.075
        for r, color in enumerate(row_colors):
            cy = gy + (nrow - 1 - r) * (ch + gap) + ch / 2
            ax.add_patch(
                Circle(
                    (dot_x, cy), width * 0.029,
                    facecolor=color, edgecolor=INK, linewidth=0.55,
                    alpha=alpha, zorder=zorder + 0.5,
                )
            )

    # Rear-to-front diagonal stack: a compact visual counterpart to the
    # successive UMAP snapshots elsewhere in Figure 1.
    draw_card(0.31, 0.31, 0.54, 0.54, alpha=0.58, zorder=1, phase=2)
    draw_card(0.23, 0.23, 0.57, 0.57, alpha=0.78, zorder=3, phase=1)
    draw_card(0.14, 0.14, 0.61, 0.61, alpha=1.00, zorder=5, phase=0)

    if label:
        ax.text(0.50, 0.965, "gene expression", ha="center", va="top",
                color=INK, fontsize=10.5)


def activation(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
               *, color: str = MID, rad: float = 0.0, lw: float = 1.7) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
            shrinkA=8, shrinkB=10, zorder=1,
        )
    )


def repression(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
               *, color: str = MID, rad: float = 0.0, lw: float = 1.7) -> None:
    """Curved regulatory edge ending in a perpendicular repression bar."""
    # Stop before the node and draw the terminal bar explicitly.
    sx, sy = start
    ex, ey = end
    vx, vy = ex - sx, ey - sy
    norm = max(np.hypot(vx, vy), 1e-8)
    ux, uy = vx / norm, vy / norm
    stop = (ex - 0.075 * ux, ey - 0.075 * uy)
    if abs(rad) < 1e-9:
        # A direct Line2D segment shares the exact terminal coordinate with
        # the bar, avoiding the small renderer gap produced by patch shrinking.
        ax.plot([sx, stop[0]], [sy, stop[1]], color=color,
                linewidth=lw, zorder=1)
    else:
        ax.add_patch(
            FancyArrowPatch(
                start, stop, arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=1,
            )
        )
    px, py = -uy, ux
    half = 0.025
    ax.plot([stop[0] - half * px, stop[0] + half * px],
            [stop[1] - half * py, stop[1] + half * py],
            color=color, linewidth=lw, zorder=2)


def tf_gene_relationship(ax: plt.Axes, *, label: bool = False) -> None:
    """Clean signed TF-to-target relationship diagram."""
    tfs = [(0.18, 0.75), (0.40, 0.79), (0.63, 0.75), (0.83, 0.80)]
    targets = [(0.14, 0.27), (0.31, 0.22), (0.49, 0.27), (0.67, 0.22), (0.85, 0.27)]

    # A compact neutral network. The rightmost TF deliberately fans to three
    # genes to make regulatory connectivity visually explicit.
    activation(ax, tfs[0], targets[0], rad=0.02, color=MID)
    activation(ax, tfs[0], targets[2], rad=-0.12, color=MID)
    activation(ax, tfs[1], targets[1], rad=0.03, color=MID)
    activation(ax, tfs[1], targets[3], rad=-0.12, color=MID)
    activation(ax, tfs[2], targets[1], rad=0.12, color=MID)
    activation(ax, tfs[2], targets[3], rad=-0.03, color=MID)
    activation(ax, tfs[3], targets[2], rad=0.14, color=MID)
    repression(ax, tfs[3], targets[3], rad=0.0, color=MID)
    activation(ax, tfs[3], targets[4], rad=-0.02, color=MID)

    for x, y in tfs:
        ax.add_patch(Circle((x, y), 0.056, facecolor=PALE,
                            edgecolor=INK, linewidth=1.15, zorder=4))

    for x, y in targets:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.052, y - 0.045), 0.104, 0.09,
                boxstyle="round,pad=0.012,rounding_size=0.022",
                facecolor=PALE, edgecolor=INK, linewidth=1.25, zorder=4,
            )
        )

    if label:
        ax.text(0.5, 0.965, "TF–gene relationships", ha="center", va="top",
                color=INK, fontsize=8.2)
        ax.text(0.5, 0.885, "TFs", ha="center", va="top", color=INK, fontsize=8.0)
        ax.text(0.5, 0.07, "target genes", ha="center", color=INK, fontsize=8.5)


def time_batched_grns(ax: plt.Axes, *, label: bool = False) -> None:
    """Time-batch × cell-type grid with one GRN expanded as an inset."""
    ax.set_xlim(0, 1.25)
    nrow, ncol = 4, 4
    gy = 0.25
    cw, ch, row_gap = 0.070, 0.105, 0.018
    # A wider gap before the final column makes room for omitted intervals.
    col_x = [0.045, 0.145, 0.245, 0.525]

    # Explicit time sequence: observed batches, skipped intervals, later batch.
    time_y = gy + nrow * (ch + row_gap) + 0.050
    time_x = [x + cw / 2 for x in col_x]
    for x in time_x:
        ax.add_patch(Circle((x, time_y), 0.015,
                            facecolor=PALE, edgecolor=INK, linewidth=0.8))
    # Four identical, freestanding arrow glyphs. Their fixed length and
    # centered placement keep them detached from dots and omission marks.
    arrow_centers = [
        (time_x[0] + time_x[1]) / 2,
        (time_x[1] + time_x[2]) / 2,
        (time_x[2] + 0.390) / 2,
        (0.434 + time_x[3]) / 2,
    ]
    arrow_length = 0.045
    for center in arrow_centers:
        ax.add_patch(FancyArrowPatch(
            (center - arrow_length / 2, time_y),
            (center + arrow_length / 2, time_y),
            arrowstyle="-|>", mutation_scale=5.5, color=MID, linewidth=1.05,
            shrinkA=0, shrinkB=0,
        ))
    for x in (0.390, 0.412, 0.434):
        ax.add_patch(Circle((x, time_y), 0.0052, facecolor=INK, edgecolor="none"))

    selected = (1, 3)  # zoom a cell from the final, post-ellipsis batch
    selected_box = None
    grid_values = np.array([
        [0.18, 0.36, 0.61, 0.82],
        [0.27, 0.52, 0.74, 0.91],
        [0.42, 0.69, 0.48, 0.76],
        [0.63, 0.44, 0.79, 0.57],
    ])
    for r in range(nrow):
        y = gy + r * (ch + row_gap)
        for c, x in enumerate(col_x):
            is_selected = (r, c) == selected
            box = FancyBboxPatch(
                (x, y), cw, ch,
                boxstyle="round,pad=0.005,rounding_size=0.014",
                facecolor=HEATMAP(grid_values[r, c]),
                edgecolor=INK if is_selected else MID,
                linewidth=2.0 if is_selected else 0.9,
                zorder=3,
            )
            ax.add_patch(box)
            if is_selected:
                selected_box = (x, y)
        # Repeat the omission mark in each cell-type row so the grid itself,
        # not only its header, communicates skipped time intervals.
        for dx in (0.390, 0.412, 0.434):
            ax.add_patch(Circle((dx, y + ch / 2), 0.0052,
                                facecolor=INK, edgecolor="none", zorder=4))

    # Expanded single-bin GRN on the right.
    ix, iy, iw, ih = 0.66, 0.22, 0.55, 0.55
    ax.add_patch(FancyBboxPatch(
        (ix, iy), iw, ih,
        boxstyle="round,pad=0.014,rounding_size=0.040",
        facecolor=WHITE, edgecolor=INK, linewidth=1.55, zorder=2,
    ))
    assert selected_box is not None
    sx, sy = selected_box
    # The two lines share the selected cell's right corners and inset's left
    # corners, making the zoom relationship explicit and unambiguous.
    ax.plot([sx + cw + 0.006, ix], [sy + ch + 0.004, iy + ih - 0.035],
            color=INK, linewidth=1.35, zorder=1)
    ax.plot([sx + cw + 0.006, ix], [sy - 0.004, iy + 0.035],
            color=INK, linewidth=1.35, zorder=1)

    # Irregular Cytoscape-like directed graph, rather than a two-level
    # TF-versus-target relationship diagram.
    nodes = [
        (ix + 0.070, iy + 0.440), (ix + 0.245, iy + 0.475),
        (ix + 0.465, iy + 0.420), (ix + 0.135, iy + 0.325),
        (ix + 0.310, iy + 0.355), (ix + 0.475, iy + 0.290),
        (ix + 0.060, iy + 0.205), (ix + 0.225, iy + 0.225),
        (ix + 0.405, iy + 0.190), (ix + 0.495, iy + 0.105),
        (ix + 0.300, iy + 0.085), (ix + 0.125, iy + 0.085),
        (ix + 0.410, iy + 0.500),
    ]
    edges = [
        (0, 1), (0, 3), (1, 3), (1, 4), (1, 12), (12, 2),
        (2, 4), (2, 5), (3, 4), (3, 6), (3, 7), (4, 5),
        (4, 7), (4, 8), (5, 8), (5, 9), (6, 7), (6, 11),
        (7, 8), (7, 10), (7, 11), (8, 9), (8, 10), (10, 9),
    ]
    node_sizes = [0.018, 0.025, 0.018, 0.021, 0.028, 0.019, 0.018,
                  0.024, 0.021, 0.017, 0.020, 0.018, 0.017]
    for i, j in edges:
        p0 = np.asarray(nodes[i], dtype=float)
        p1 = np.asarray(nodes[j], dtype=float)
        unit = (p1 - p0) / np.linalg.norm(p1 - p0)
        edge_start = p0 + unit * node_sizes[i]
        edge_end = p1 - unit * (node_sizes[j] + 0.003)
        ax.add_patch(FancyArrowPatch(
            edge_start, edge_end, arrowstyle="-|>", mutation_scale=6.0,
            color=MID, linewidth=1.0, shrinkA=0, shrinkB=0, zorder=3,
        ))
    for (x, y), radius in zip(nodes, node_sizes):
        ax.add_patch(Circle((x, y), radius, facecolor=PALE,
                            edgecolor=INK, linewidth=0.85, zorder=4))

    if label:
        ax.text(0.50, 0.965, "time-batched GRNs", ha="center", va="top",
                color=INK, fontsize=9.0)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(path.with_suffix(".svg"), **common)
    fig.savefig(path.with_suffix(".pdf"), **common)
    fig.savefig(path.with_suffix(".png"), dpi=600, **common)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    setup()

    fig, ax = blank_ax((2.15, 2.15))
    expression_matrix(ax)
    save(fig, args.out_dir / "gene_expression_matrix_icon")

    fig, ax = blank_ax((2.55, 2.15))
    tf_gene_relationship(ax)
    save(fig, args.out_dir / "tf_gene_relationship_icon")

    fig, ax = blank_ax((3.45, 2.20))
    time_batched_grns(ax)
    save(fig, args.out_dir / "time_batched_grn_icon")

    # A labeled preview helps select/position the assets; the two files above
    # remain label-free for integration into the architecture composition.
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.55))
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
    expression_matrix(axes[0], label=True)
    tf_gene_relationship(axes[1], label=True)
    time_batched_grns(axes[2], label=True)
    fig.subplots_adjust(wspace=0.12)
    save(fig, args.out_dir / "expression_and_grn_icons_preview")


if __name__ == "__main__":
    main()
