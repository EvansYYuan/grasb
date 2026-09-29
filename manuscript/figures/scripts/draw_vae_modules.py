#!/usr/bin/env python
"""Draw paper-ready frozen encoder and decoder modules for Figure 1.

The artwork uses only Matplotlib vector primitives and is inspired by the
layered topology of the CC0 ``variational-autoencoder.svg`` reference. It is
redrawn in the manuscript's muted visual grammar rather than copied directly.

Run:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/draw_vae_modules.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"

TEAL = "#3FA89B"
TEAL_LIGHT = "#DDEDEE"
INDIGO = "#6762B5"
INK = "#626A6D"
EDGE = "#A2AAAD"
NODE = "#EEF0EF"
WHITE = "#FFFFFF"


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def blank_ax(figsize: tuple[float, float] = (3.25, 1.85)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def layer_positions(x: float, count: int, spread: float) -> list[tuple[float, float]]:
    if count == 1:
        return [(x, 0.50)]
    ys = np.linspace(0.50 + spread / 2, 0.50 - spread / 2, count)
    return [(x, float(y)) for y in ys]


def connect_layers(ax: plt.Axes, left: list[tuple[float, float]],
                   right: list[tuple[float, float]]) -> None:
    """Dense enough to read as a neural layer, sparse enough for print."""
    for i, p0 in enumerate(left):
        # Connect each source to its two or three nearest destination nodes.
        order = np.argsort([abs(p0[1] - p1[1]) for p1 in right])
        degree = min(3 if len(left) <= 4 else 2, len(right))
        for j in order[:degree]:
            p1 = right[int(j)]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=EDGE, linewidth=1.15, alpha=0.82, zorder=1)


def neural_module(ax: plt.Axes, *, decoder: bool, lock: bool) -> None:
    """Draw a contracting encoder or expanding decoder."""
    counts = [5, 4, 3, 2]
    spreads = [0.64, 0.52, 0.39, 0.24]
    if decoder:
        counts = counts[::-1]
        spreads = spreads[::-1]

    xs = [0.11, 0.37, 0.63, 0.89]
    layers = [layer_positions(x, n, spread)
              for x, n, spread in zip(xs, counts, spreads)]

    # Directional trapezoids communicate contraction/expansion without the
    # confusing fabric-like funnels in the AI reference.
    if decoder:
        boundary = [(-0.020, 0.33), (-0.020, 0.67),
                    (1.020, 0.94), (1.020, 0.06)]
    else:
        boundary = [(-0.020, 0.06), (-0.020, 0.94),
                    (1.020, 0.67), (1.020, 0.33)]
    ax.add_patch(Polygon(
        boundary, closed=True, facecolor=TEAL_LIGHT, edgecolor=TEAL,
        linewidth=1.55, alpha=0.55, zorder=0,
    ))

    for left, right in zip(layers[:-1], layers[1:]):
        connect_layers(ax, left, right)

    for layer_i, layer in enumerate(layers):
        is_latent = (layer_i == 3 and not decoder) or (layer_i == 0 and decoder)
        is_gene_space = (layer_i == 0 and not decoder) or (layer_i == 3 and decoder)
        for x, y in layer:
            ax.add_patch(Circle(
                (x, y), 0.038 if is_latent else 0.033,
                facecolor=INDIGO if is_latent else (TEAL if is_gene_space else NODE),
                edgecolor=INK, linewidth=1.15, zorder=4,
            ))


def save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.02, transparent=True)
    fig.savefig(base.with_suffix(".svg"), **common)
    fig.savefig(base.with_suffix(".pdf"), **common)
    fig.savefig(base.with_suffix(".png"), dpi=600, **common)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    setup()

    for name, decoder in (("encoder", False), ("decoder", True)):
        for with_lock in (True, False):
            fig, ax = blank_ax()
            neural_module(ax, decoder=decoder, lock=with_lock)
            suffix = "" if with_lock else "_no_lock"
            save(fig, args.out_dir / f"frozen_{name}_module{suffix}")

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.35))
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor("none")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
    neural_module(axes[0], decoder=False, lock=True)
    neural_module(axes[1], decoder=True, lock=True)
    axes[0].text(0.5, 0.98, "frozen encoder", ha="center", va="top",
                 color=INK, fontsize=10)
    axes[1].text(0.5, 0.98, "frozen decoder", ha="center", va="top",
                 color=INK, fontsize=10)
    fig.subplots_adjust(wspace=0.10)
    save(fig, args.out_dir / "frozen_encoder_decoder_preview")


if __name__ == "__main__":
    main()
