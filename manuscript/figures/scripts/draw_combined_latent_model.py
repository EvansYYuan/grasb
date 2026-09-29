#!/usr/bin/env python
"""Compose encoder, latent uSB, decoder, and GRN Jacobian pushforward."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from draw_latent_usb_module import draw_module as draw_usb
from draw_vae_modules import neural_module as draw_neural_module


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"
CORAL = "#F06455"
INK = "#626A6D"
TEAL = "#3FA89B"
INDIGO = "#34358D"


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def compose(*, labels: bool) -> plt.Figure:
    fig = plt.figure(figsize=(12.2, 3.65))
    fig.patch.set_alpha(0)

    encoder_ax = fig.add_axes([0.015, 0.16, 0.235, 0.72])
    bridge_ax = fig.add_axes([0.225, 0.12, 0.565, 0.78])
    decoder_ax = fig.add_axes([0.765, 0.16, 0.235, 0.72])

    for ax in (encoder_ax, decoder_ax):
        ax.set_facecolor("none")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.axis("off")
    bridge_ax.set_facecolor("none")
    bridge_ax.set_xlim(0, 1.60)
    bridge_ax.set_ylim(0, 0.92)
    bridge_ax.set_aspect("equal")
    bridge_ax.axis("off")

    draw_neural_module(encoder_ax, decoder=False, lock=False)
    draw_usb(bridge_ax, labels=False)
    draw_neural_module(decoder_ax, decoder=True, lock=False)

    # State flow enters the encoder independently of GRN guidance.
    fig.add_artist(FancyArrowPatch(
        (0.002, 0.62), (0.055, 0.62), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, color=TEAL,
        linewidth=2.0, zorder=20,
    ))
    fig.add_artist(FancyArrowPatch(
        (0.232, 0.535), (0.318, 0.535), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, color=INDIGO,
        linewidth=2.1, zorder=19,
    ))
    fig.add_artist(FancyArrowPatch(
        (0.695, 0.535), (0.775, 0.535), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, color=INDIGO,
        linewidth=2.1, zorder=19,
    ))

    # The coral tangent changes orientation across the encoder: an explicit
    # visual representation of v_z = J_E(x)b_GRN(x,t), not a second state path.
    fig.add_artist(FancyArrowPatch(
        (0.012, 0.105), (0.082, 0.285), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=11, color=CORAL,
        linewidth=2.2, zorder=20,
    ))
    fig.add_artist(FancyArrowPatch(
        (0.082, 0.285), (0.205, 0.355), transform=fig.transFigure,
        arrowstyle="-", color=CORAL, linewidth=1.7,
        linestyle=(0, (2.2, 2.5)), zorder=20,
    ))
    fig.add_artist(FancyArrowPatch(
        (0.205, 0.355), (0.300, 0.535), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=12, color=CORAL,
        linewidth=2.4, connectionstyle="arc3,rad=-0.12", zorder=21,
    ))

    if labels:
        fig.text(0.132, 0.94, "frozen encoder", ha="center", va="top",
                 color=INK, fontsize=12)
        fig.text(0.505, 0.95, "latent unbalanced Schrödinger bridge",
                 ha="center", va="top", color=INK, fontsize=12)
        fig.text(0.882, 0.94, "frozen decoder", ha="center", va="top",
                 color=INK, fontsize=12)
        fig.text(0.018, 0.080, r"$b_{\mathrm{GRN}}(x,t)$", color=CORAL,
                 fontsize=10, ha="left", va="center")
        fig.text(0.218, 0.355, r"$J_E(x)b_{\mathrm{GRN}}(x,t)$", color=CORAL,
                 fontsize=10, ha="left", va="center")
        fig.text(0.004, 0.665, r"$x$", color=TEAL, fontsize=11,
                 ha="left", va="center")

    return fig


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
    save(compose(labels=False), args.out_dir / "combined_encoder_usb_decoder")
    save(compose(labels=True), args.out_dir / "combined_encoder_usb_decoder_labeled")


if __name__ == "__main__":
    main()
