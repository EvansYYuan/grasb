#!/usr/bin/env python
"""Draw a paper-ready latent unbalanced Schrödinger bridge module.

The icon is intentionally explicit about the OT/SB semantics: source,
intermediate, and target marginals anchor stochastic paths; a thick path shows
mean transport; an amber channel shows mass variation; and a short coral vector
shows local GRN guidance without becoming the generated trajectory.

Run:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/draw_latent_usb_module.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"

INDIGO = "#6662B7"
DEEP_INDIGO = "#34358D"
LAVENDER = "#E9E4FA"
LAVENDER_EDGE = "#B8ACEA"
PATH_PURPLE = "#A79CE5"
CORAL = "#F06455"
INK = "#626A6D"
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


def blank_ax() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.4, 2.45))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1.60)
    ax.set_ylim(0, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def wave_wrapper(ax: plt.Axes) -> None:
    """Open wave with both fill and boundaries fading at either end."""
    x = np.linspace(0.06, 1.54, 85)
    phase = (x - 0.06) / (1.54 - 0.06)
    top = 0.69 + 0.055 * np.sin(2.35 * np.pi * phase + 0.15)
    bottom = 0.25 + 0.052 * np.sin(2.35 * np.pi * phase - 0.65)
    fade = np.sin(np.pi * phase) ** 1.35
    for i in range(len(x) - 1):
        # Using the minimum rather than the segment average makes the outermost
        # strips exactly transparent, eliminating visible vertical end walls.
        alpha = 0.52 * float(min(fade[i], fade[i + 1]))
        ax.fill_between(x[i:i + 2], bottom[i:i + 2], top[i:i + 2],
                        facecolor=LAVENDER, edgecolor="none", alpha=alpha,
                        zorder=0)
        ax.plot(x[i:i + 2], top[i:i + 2], color=LAVENDER_EDGE,
                linewidth=1.2, alpha=0.70 * alpha / 0.50, zorder=0.5)
        ax.plot(x[i:i + 2], bottom[i:i + 2], color=LAVENDER_EDGE,
                linewidth=1.2, alpha=0.70 * alpha / 0.50, zorder=0.5)


def marginal(center: tuple[float, float], n: int, scale: tuple[float, float],
             rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(n, 2)) * np.asarray(scale) + np.asarray(center)
    return points


def bridge_path(ax: plt.Axes, p0: np.ndarray, p1: np.ndarray,
                bend1: float, bend2: float, alpha: float) -> tuple[np.ndarray, ...]:
    """Cubic stochastic trajectory anchored in source and target marginals."""
    verts = [
        tuple(p0),
        (0.62, p0[1] + bend1),
        (1.02, p1[1] + bend2),
        tuple(p1),
    ]
    codes = [MplPath.MOVETO] + [MplPath.CURVE4] * 3
    ax.add_patch(PathPatch(
        MplPath(verts, codes), facecolor="none", edgecolor=PATH_PURPLE,
        linewidth=0.62, alpha=alpha, zorder=1,
    ))
    return tuple(np.asarray(v, dtype=float) for v in verts)


def bezier_point(control: tuple[np.ndarray, ...], t: float) -> np.ndarray:
    p0, p1, p2, p3 = control
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def draw_module(ax: plt.Axes, *, labels: bool) -> None:
    rng = np.random.default_rng(41)
    wave_wrapper(ax)

    centers = [(0.24, 0.48), (1.36, 0.49)]
    clouds = [
        marginal(centers[0], 78, (0.050, 0.105), rng),
        marginal(centers[1], 82, (0.052, 0.108), rng),
    ]
    for cloud in clouds:
        cloud[:, 1] = np.clip(cloud[:, 1], 0.285, 0.665)

    # Stochastic bridge paths are explicitly pinned to particles in each
    # marginal, making this read as a bridge rather than decorative dynamics.
    controls = []
    for k in range(34):
        p0 = clouds[0][rng.integers(len(clouds[0]))]
        p1 = clouds[1][rng.integers(len(clouds[1]))]
        controls.append(bridge_path(
            ax, p0, p1,
            bend1=rng.uniform(-0.17, 0.17),
            bend2=rng.uniform(-0.17, 0.17),
            alpha=rng.uniform(0.22, 0.42),
        ))

    # Dominant barycentric transport remains visually distinct from the path
    # samples and from the short coral biological-prior vector.
    ax.add_patch(FancyArrowPatch(
        (0.22, 0.50), (1.38, 0.50),
        arrowstyle="-|>", mutation_scale=12,
        connectionstyle="arc3,rad=0.055",
        color=DEEP_INDIGO, linewidth=2.5, zorder=3,
        shrinkA=7, shrinkB=7,
    ))

    # Draw marginal particles last so every path visibly terminates at a cloud.
    for cloud_i, cloud in enumerate(clouds):
        sizes = rng.uniform(0.008, 0.013, len(cloud))
        for (x, y), radius in zip(cloud, sizes):
            ax.add_patch(Circle(
                (x, y), radius,
                facecolor=INDIGO, edgecolor=WHITE,
                linewidth=0.30, alpha=0.88, zorder=4,
            ))
        # A larger central particle visually anchors the mean transport.
        cx, cy = centers[cloud_i]
        ax.add_patch(Circle((cx, cy), 0.027, facecolor=INDIGO,
                            edgecolor=INK, linewidth=0.8, zorder=5))

    # Following Pariset et al. Fig. 4e, mass change is attached directly to
    # continuous trajectories: filled dots denote death and hollow dots birth.
    event_specs = [
        (2, 0.30, "death"), (8, 0.46, "birth"),
        (14, 0.58, "death"), (21, 0.69, "birth"),
        (27, 0.79, "death"), (31, 0.38, "birth"),
    ]
    for path_i, t, event in event_specs:
        x, y = bezier_point(controls[path_i], t)
        ax.add_patch(Circle(
            (x, y), 0.012,
            facecolor=INK if event == "death" else WHITE,
            edgecolor=INK, linewidth=0.85, zorder=6,
        ))

    if labels:
        for x, text in zip([0.24, 1.36], [r"$\rho_0$", r"$\rho_T$"]):
            ax.text(x, 0.735, text, ha="center", va="center",
                    fontsize=10, color=INK)
        ax.text(0.80, 0.885, "latent unbalanced Schrödinger bridge",
                ha="center", va="top", fontsize=11, color=INK)
        ax.add_patch(Circle((0.65, 0.165), 0.010, facecolor=INK,
                            edgecolor=INK, linewidth=0.8, zorder=6))
        ax.text(0.67, 0.165, "death", ha="left", va="center",
                fontsize=8.0, color=INK)
        ax.add_patch(Circle((0.88, 0.165), 0.010, facecolor=WHITE,
                            edgecolor=INK, linewidth=0.8, zorder=6))
        ax.text(0.90, 0.165, "birth", ha="left", va="center",
                fontsize=8.0, color=INK)


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

    fig, ax = blank_ax()
    draw_module(ax, labels=False)
    save(fig, args.out_dir / "latent_unbalanced_sb_module")

    fig, ax = blank_ax()
    draw_module(ax, labels=True)
    save(fig, args.out_dir / "latent_unbalanced_sb_module_labeled")


if __name__ == "__main__":
    main()
