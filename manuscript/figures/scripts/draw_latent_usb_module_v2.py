#!/usr/bin/env python
"""Draw V2 of the latent unbalanced Schrödinger bridge module.

V2 uses three dense marginals, visibly stochastic Brownian-style paths, no
central mean-flow curve, and a compact three-state mass profile.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"

INDIGO = "#6662B7"
LAVENDER = "#E9E4FA"
LAVENDER_EDGE = "#B8ACEA"
PATH_PURPLE = "#968BD8"
AMBER = "#DFA329"
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
    fig, ax = plt.subplots(figsize=(5.5, 2.55))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1.60)
    ax.set_ylim(0, 0.94)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def open_wave(ax: plt.Axes) -> None:
    x = np.linspace(0.055, 1.545, 90)
    phase = (x - x.min()) / (x.max() - x.min())
    top = 0.735 + 0.050 * np.sin(2.35 * np.pi * phase + 0.12)
    bottom = 0.255 + 0.048 * np.sin(2.35 * np.pi * phase - 0.68)
    fade = np.sin(np.pi * phase) ** 1.35

    # A clipped RGBA gradient gives genuinely open, fading ends without the
    # faint vertical walls produced by many adjacent filled polygons.
    polygon = np.column_stack([
        np.r_[x, x[::-1]],
        np.r_[top, bottom[::-1]],
    ])
    clip = PathPatch(MplPath(polygon, closed=True), facecolor="none",
                     edgecolor="none", transform=ax.transData)
    ax.add_patch(clip)
    rgba = np.zeros((2, 640, 4), dtype=float)
    rgb = mpl.colors.to_rgb(LAVENDER)
    rgba[..., :3] = rgb
    u = np.linspace(0.0, 1.0, rgba.shape[1])
    # Keep a short fully transparent plateau at each clipped end so raster
    # antialiasing cannot reveal the otherwise invisible vertical clip edge.
    inner = np.clip((u - 0.055) / 0.89, 0.0, 1.0)
    rgba[..., 3] = 0.50 * (np.sin(np.pi * inner) ** 1.35)[None, :]
    image = ax.imshow(rgba, extent=(x.min(), x.max(), 0.18, 0.82),
                      origin="lower", interpolation="bilinear",
                      aspect="auto", zorder=0)
    image.set_clip_path(clip)

    for i in range(len(x) - 1):
        boundary_alpha = 0.68 * float(min(fade[i], fade[i + 1]))
        ax.plot(x[i:i + 2], top[i:i + 2], color=LAVENDER_EDGE,
                linewidth=1.15, alpha=boundary_alpha, zorder=0.5)
        ax.plot(x[i:i + 2], bottom[i:i + 2], color=LAVENDER_EDGE,
                linewidth=1.15, alpha=boundary_alpha, zorder=0.5)


def cloud(center: tuple[float, float], n: int, scale: tuple[float, float],
          rng: np.random.Generator) -> np.ndarray:
    xy = rng.normal(size=(n, 2)) * np.asarray(scale) + np.asarray(center)
    xy[:, 1] = np.clip(xy[:, 1], 0.315, 0.700)
    return xy


def brownian_segment(p0: np.ndarray, p1: np.ndarray, n: int,
                     amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """A discrete Brownian bridge with noise constrained to vanish at ends."""
    t = np.linspace(0.0, 1.0, n)
    baseline = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
    increments = rng.normal(scale=1.0, size=(n, 2))
    walk = np.cumsum(increments, axis=0)
    bridge = walk - t[:, None] * walk[-1][None, :]
    bridge -= bridge[0][None, :]
    bridge[:, 0] *= 0.010
    bridge[:, 1] *= amplitude
    return baseline + bridge


def draw_mass_profile(ax: plt.Axes, centers: list[tuple[float, float]]) -> None:
    """Three observed mass states joined by one continuous amber profile."""
    x = np.linspace(centers[0][0], centers[-1][0], 140)
    # Initial mass, intermediate expansion, and reduced terminal mass.
    knot_x = np.array([centers[0][0], centers[1][0], centers[2][0]])
    knot_y = np.array([0.170, 0.215, 0.145])
    y = np.interp(x, knot_x, knot_y)
    # Smooth the two piecewise-linear segments without adding timepoints.
    kernel = np.hanning(21)
    kernel /= kernel.sum()
    padded = np.pad(y, (10, 10), mode="edge")
    y = np.convolve(padded, kernel, mode="valid")
    ax.plot(x, y, color=AMBER, linewidth=1.45, zorder=3)
    radii = [0.014, 0.025, 0.011]
    for (cx, _), cy, radius in zip(centers, knot_y, radii):
        ax.add_patch(Circle((cx, cy), radius, facecolor=AMBER,
                            edgecolor=WHITE, linewidth=0.45, zorder=4))


def draw_module(ax: plt.Axes, *, labels: bool) -> None:
    rng = np.random.default_rng(73)
    open_wave(ax)

    centers = [(0.235, 0.505), (0.800, 0.520), (1.365, 0.500)]
    clouds = [
        cloud(centers[0], 76, (0.050, 0.100), rng),
        cloud(centers[1], 62, (0.055, 0.105), rng),
        cloud(centers[2], 82, (0.052, 0.105), rng),
    ]

    # Every path passes through an actual intermediate-marginal particle.
    for _ in range(42):
        p0 = clouds[0][rng.integers(len(clouds[0]))]
        pm = clouds[1][rng.integers(len(clouds[1]))]
        p1 = clouds[2][rng.integers(len(clouds[2]))]
        left = brownian_segment(p0, pm, 18, rng.uniform(0.010, 0.018), rng)
        right = brownian_segment(pm, p1, 18, rng.uniform(0.010, 0.018), rng)
        path = np.vstack([left, right[1:]])
        ax.plot(path[:, 0], path[:, 1], color=PATH_PURPLE,
                linewidth=0.52, alpha=rng.uniform(0.20, 0.38), zorder=1)

    # Dense marginals remain primary; paths are drawn first and terminate under
    # the particles, emphasizing the bridge constraints.
    for cloud_xy in clouds:
        radii = rng.uniform(0.0075, 0.0120, len(cloud_xy))
        for (x, y), radius in zip(cloud_xy, radii):
            ax.add_patch(Circle((x, y), radius, facecolor=INDIGO,
                                edgecolor=WHITE, linewidth=0.28,
                                alpha=0.88, zorder=4))

    draw_mass_profile(ax, centers)

    if labels:
        for x, text in zip([c[0] for c in centers],
                           [r"$\rho_0$", r"$\rho_t$", r"$\rho_T$"]):
            ax.text(x, 0.775, text, ha="center", va="center",
                    fontsize=9.5, color=INK)
        ax.text(0.80, 0.920, "latent unbalanced Schrödinger bridge",
                ha="center", va="top", fontsize=10.5, color=INK)
        ax.text(1.405, 0.135, "relative mass", ha="left", va="center",
                fontsize=8.0, color=AMBER)


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
    save(fig, args.out_dir / "latent_unbalanced_sb_module_v2")

    fig, ax = blank_ax()
    draw_module(ax, labels=True)
    save(fig, args.out_dir / "latent_unbalanced_sb_module_v2_labeled")


if __name__ == "__main__":
    main()
