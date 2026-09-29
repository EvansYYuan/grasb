#!/usr/bin/env python
"""Draw a compact, signed drift-gene ranking icon."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"


def blend(a: str, b: str, n: int) -> list[tuple[float, float, float]]:
    c0, c1 = np.array(mpl.colors.to_rgb(a)), np.array(mpl.colors.to_rgb(b))
    return [tuple(c0 * (1 - t) + c1 * t) for t in np.linspace(0, 1, n)]


def draw(out_dir: Path) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    # Ranked illustrative scores: direction and ordering are the message;
    # specific gene names and empirical magnitudes belong in the data panel.
    positive = np.array([1.00, 0.86, 0.67, 0.56, 0.48, 0.42,
                         0.36, 0.30, 0.25, 0.20, 0.16, 0.12, 0.09])
    negative = -np.array([0.10, 0.20, 0.30, 0.39, 0.49, 0.60, 0.72, 0.86])
    values = np.r_[positive, negative]
    x = np.arange(len(values))

    pos_colors = blend("#F06455", "#E7A52B", len(positive))
    neg_colors = blend("#A9B9EE", "#6662B7", len(negative))

    fig, ax = plt.subplots(figsize=(4.15, 2.20))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.bar(x[:len(positive)], positive, width=0.66,
           color=pos_colors, edgecolor=pos_colors, linewidth=0.65, zorder=3)
    ax.bar(x[len(positive):], negative, width=0.66,
           color=neg_colors, edgecolor=neg_colors, linewidth=0.65, zorder=3)

    axis_x = -1.20
    # Leave a deliberate stretch of bare baseline after the final bar before
    # the arrowhead; otherwise the final ranked gene visually merges with it.
    axis_tip = len(values) + 0.95
    head_length, head_half_height = 0.48, 0.034
    ax.plot([axis_x, axis_tip - head_length], [0, 0], color="#000000",
            linewidth=1.15, solid_capstyle="butt", zorder=4)
    ax.add_patch(Polygon([
        (axis_tip, 0),
        (axis_tip - head_length, head_half_height),
        (axis_tip - head_length, -head_half_height),
    ], closed=True, facecolor="#000000", edgecolor="#000000", zorder=5))
    ax.plot([axis_x, axis_x], [-0.94, 1.08], color="#000000",
            linewidth=1.15, solid_capstyle="butt", zorder=4)
    ax.text(axis_x - 0.78, 1.02, "+", ha="center", va="center",
            fontsize=10, color="#000000")
    ax.text(axis_x - 0.78, -0.88, "−", ha="center", va="center",
            fontsize=10, color="#000000")

    ax.set_xlim(-2.25, len(values) + 1.05)
    ax.set_ylim(-1.00, 1.12)
    ax.axis("off")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "drift_gene_ranking_icon"
    common = dict(bbox_inches="tight", pad_inches=0.025, transparent=True)
    fig.savefig(base.with_suffix(".svg"), **common)
    fig.savefig(base.with_suffix(".pdf"), **common)
    fig.savefig(base.with_suffix(".png"), dpi=600, **common)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    draw(args.out_dir)


if __name__ == "__main__":
    main()
