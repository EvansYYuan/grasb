#!/usr/bin/env python
"""Draw V3 latent uSB artwork with two explanatory subpanels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = PROJECT / "manuscript/figures/components"

INDIGO = "#6662B7"
DEEP_INDIGO = "#34358D"
LAVENDER = "#E9E4FA"
LAVENDER_EDGE = "#B8ACEA"
PATH_PURPLE = "#968BD8"
CORAL = "#F06455"
AMBER = "#DFA329"
INK = "#626A6D"
MID = "#9AA2A5"
WHITE = "#FFFFFF"


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Libertinus Serif", "STIX Two Text", "DejaVu Serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def blank_ax() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.7, 3.25))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.set_xlim(0, 1.60)
    ax.set_ylim(0, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def open_wave(ax: plt.Axes) -> None:
    x = np.linspace(0.055, 1.545, 90)
    phase = (x - x.min()) / (x.max() - x.min())
    top = 1.025 + 0.047 * np.sin(2.35 * np.pi * phase + 0.12)
    bottom = 0.535 + 0.046 * np.sin(2.35 * np.pi * phase - 0.68)
    fade = np.sin(np.pi * phase) ** 1.35
    polygon = np.column_stack([np.r_[x, x[::-1]], np.r_[top, bottom[::-1]]])
    clip = PathPatch(MplPath(polygon, closed=True), facecolor="none",
                     edgecolor="none", transform=ax.transData)
    ax.add_patch(clip)
    rgba = np.zeros((2, 640, 4), dtype=float)
    rgba[..., :3] = mpl.colors.to_rgb(LAVENDER)
    u = np.linspace(0, 1, rgba.shape[1])
    inner = np.clip((u - 0.055) / 0.89, 0, 1)
    rgba[..., 3] = 0.50 * (np.sin(np.pi * inner) ** 1.35)[None, :]
    im = ax.imshow(rgba, extent=(x.min(), x.max(), 0.46, 1.10),
                   origin="lower", interpolation="bilinear", aspect="auto", zorder=0)
    im.set_clip_path(clip)
    for i in range(len(x) - 1):
        a = 0.68 * float(min(fade[i], fade[i + 1]))
        ax.plot(x[i:i+2], top[i:i+2], color=LAVENDER_EDGE,
                linewidth=1.15, alpha=a, zorder=0.5)
        ax.plot(x[i:i+2], bottom[i:i+2], color=LAVENDER_EDGE,
                linewidth=1.15, alpha=a, zorder=0.5)


def cloud(center: tuple[float, float], n: int, scale: tuple[float, float],
          rng: np.random.Generator) -> np.ndarray:
    xy = rng.normal(size=(n, 2)) * np.asarray(scale) + np.asarray(center)
    xy[:, 1] = np.clip(xy[:, 1], 0.590, 0.990)
    return xy


def brownian_segment(p0: np.ndarray, p1: np.ndarray, n: int,
                     amplitude: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, n)
    baseline = p0[None, :] * (1 - t[:, None]) + p1[None, :] * t[:, None]
    walk = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    bridge = walk - t[:, None] * walk[-1][None, :]
    bridge -= bridge[0][None, :]
    bridge[:, 0] *= 0.010
    bridge[:, 1] *= amplitude
    return baseline + bridge


def panel(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=WHITE, edgecolor=MID, linewidth=1.0,
        linestyle=(0, (3, 2.4)), alpha=0.96, zorder=8,
    ))


def mass_panel(ax: plt.Axes, *, labels: bool) -> None:
    x0, y0, w, h = 0.20, 0.075, 0.49, 0.285
    panel(ax, x0, y0, w, h)
    axis_x, axis_y = x0 + 0.065, y0 + 0.055
    axis_right, axis_top = x0 + w - 0.035, y0 + h - 0.035
    # A clean black L forms the axes. Each shaft ends at the base of a true
    # triangular arrowhead, so no line protrudes through the head.
    axis_color = "#000000"
    head_length, head_half_width = 0.024, 0.012
    ax.plot([axis_x, axis_right - head_length], [axis_y, axis_y], color=axis_color,
            linewidth=1.25, zorder=10)
    ax.plot([axis_x, axis_x], [axis_y, axis_top - head_length], color=axis_color,
            linewidth=1.25, zorder=10)
    ax.add_patch(Polygon([
        (axis_right, axis_y),
        (axis_right - head_length, axis_y + head_half_width),
        (axis_right - head_length, axis_y - head_half_width),
    ], closed=True, facecolor=axis_color, edgecolor=axis_color, zorder=11))
    ax.add_patch(Polygon([
        (axis_x, axis_top),
        (axis_x - head_half_width, axis_top - head_length),
        (axis_x + head_half_width, axis_top - head_length),
    ], closed=True, facecolor=axis_color, edgecolor=axis_color, zorder=11))

    xs = np.array([x0 + 0.125, x0 + 0.265, x0 + 0.405])
    ys = np.array([y0 + 0.115, y0 + 0.205, y0 + 0.095])
    dense_x = np.linspace(xs[0], xs[-1], 100)
    dense_y = np.interp(dense_x, xs, ys)
    kernel = np.hanning(15); kernel /= kernel.sum()
    dense_y = np.convolve(np.pad(dense_y, (7, 7), mode="edge"), kernel, mode="valid")
    ax.plot(dense_x, dense_y, color=AMBER, linewidth=1.5, zorder=10)
    for x, y, r in zip(xs, ys, [0.013, 0.027, 0.010]):
        ax.add_patch(Circle((x, y), r, facecolor=AMBER,
                            edgecolor=WHITE, linewidth=0.4, zorder=11))


def guidance_panel(ax: plt.Axes, *, labels: bool) -> None:
    x0, y0, w, h = 0.89, 0.075, 0.51, 0.285
    panel(ax, x0, y0, w, h)
    # Two input vector fields interfere to form a third resultant field:
    #       SB drift  +  Jacobian-mapped GRN drift  ->  shaped drift.
    # Dense 2-D quivers, with direction and magnitude changing by location,
    # distinguish these from a collection of trajectories.
    def vector_field(cx: float, color: str, kind: str) -> None:
        gx = np.linspace(-1, 1, 5)
        gy = np.linspace(-1, 1, 6)
        for yy in gy:
            for xx in gx:
                # Baseline SB field and Jacobian-mapped GRN field. The third
                # field is their pointwise weighted sum.
                sb = np.array([0.82 + 0.10 * yy, 0.24 * xx - 0.12 * yy])
                grn = np.array([-0.34 * yy + 0.12, 0.62 * xx + 0.12 * yy])
                # Strong GRN coupling makes the resultant visibly inherit the
                # GRN field's turning pattern without erasing the SB drift.
                vec = sb if kind == "sb" else grn if kind == "grn" else sb + 1.10 * grn
                norm = np.linalg.norm(vec)
                length = 0.019 + 0.007 * min(norm, 1.25) / 1.25
                delta = vec / max(norm, 1e-8) * length
                px = cx + 0.044 * xx
                py = y0 + 0.145 + 0.068 * yy
                ax.add_patch(FancyArrowPatch(
                    (px - 0.42 * delta[0], py - 0.42 * delta[1]),
                    (px + 0.58 * delta[0], py + 0.58 * delta[1]),
                    arrowstyle="-|>", mutation_scale=2.15,
                    color=color, linewidth=0.58, alpha=0.86, zorder=11,
                    shrinkA=0, shrinkB=0,
                ))

    vector_field(x0 + 0.075, PATH_PURPLE, "sb")
    vector_field(x0 + 0.235, CORAL, "grn")
    vector_field(x0 + 0.425, DEEP_INDIGO, "result")

    ax.text(x0 + 0.155, y0 + 0.145, "+", ha="center", va="center",
            fontsize=10, color=INK, fontweight="bold", zorder=12)
    ax.add_patch(FancyArrowPatch(
        (x0 + 0.305, y0 + 0.145), (x0 + 0.350, y0 + 0.145),
        arrowstyle="-|>", mutation_scale=7.5, color=INK,
        linewidth=1.1, zorder=12,
    ))
    if labels:
        ax.text(x0 + w/2, y0 + h - 0.018, "soft GRN drift shaping",
                ha="center", va="top", fontsize=5.5, color=INK, zorder=12)


def guidance_panel_proposal(ax: plt.Axes) -> None:
    """Non-additive visual: GRN guidance bends the SB-to-learned-field path."""
    x0, y0, w, h = 0.12, 0.10, 1.36, 0.62
    panel(ax, x0, y0, w, h)

    def field(cx: float, cy: float, color: str, kind: str,
              sx: float = 0.115, sy: float = 0.135) -> None:
        for yy in np.linspace(-1, 1, 6):
            for xx in np.linspace(-1, 1, 5):
                sb = np.array([0.82 + 0.10 * yy, 0.24 * xx - 0.12 * yy])
                grn = np.array([-0.34 * yy + 0.12, 0.62 * xx + 0.12 * yy])
                vec = sb if kind == "sb" else grn if kind == "grn" else sb + 1.10 * grn
                norm = np.linalg.norm(vec)
                length = 0.036 + 0.012 * min(norm, 1.25) / 1.25
                delta = vec / max(norm, 1e-8) * length
                px, py = cx + sx * xx, cy + sy * yy
                ax.add_patch(FancyArrowPatch(
                    (px - 0.42 * delta[0], py - 0.42 * delta[1]),
                    (px + 0.58 * delta[0], py + 0.58 * delta[1]),
                    arrowstyle="-|>", mutation_scale=3.4,
                    color=color, linewidth=0.72, alpha=0.88,
                    shrinkA=0, shrinkB=0, zorder=11,
                ))

    # Primary reading direction: baseline SB field becomes a learned field.
    field(x0 + 0.245, y0 + 0.385, PATH_PURPLE, "sb")
    field(x0 + 1.105, y0 + 0.385, DEEP_INDIGO, "result")
    ax.add_patch(FancyArrowPatch(
        (x0 + 0.405, y0 + 0.385), (x0 + 0.930, y0 + 0.385),
        arrowstyle="-|>", mutation_scale=10, color=INK,
        linewidth=1.35, shrinkA=0, shrinkB=0, zorder=10,
    ))

    # The GRN field enters as guidance, not as an additive operand.
    field(x0 + 0.660, y0 + 0.135, CORAL, "grn", sx=0.090, sy=0.080)
    ax.add_patch(FancyArrowPatch(
        (x0 + 0.735, y0 + 0.215), (x0 + 0.690, y0 + 0.385),
        arrowstyle="-|>", mutation_scale=9, color=CORAL,
        linewidth=1.35, connectionstyle="arc3,rad=-0.22",
        shrinkA=0, shrinkB=0, zorder=12,
    ))


def standalone_vector_field(ax: plt.Axes, kind: str) -> None:
    """Draw one frameless vector-field asset on the standard canvas."""
    colors = {"sb": PATH_PURPLE, "grn": CORAL, "result": DEEP_INDIGO}
    color = colors[kind]
    cx, cy = 0.80, 0.59
    for yy in np.linspace(-1, 1, 6):
        for xx in np.linspace(-1, 1, 5):
            sb = np.array([0.82 + 0.10 * yy, 0.24 * xx - 0.12 * yy])
            grn = np.array([-0.34 * yy + 0.12, 0.62 * xx + 0.12 * yy])
            vec = sb if kind == "sb" else grn if kind == "grn" else sb + 1.10 * grn
            norm = np.linalg.norm(vec)
            length = 0.075 + 0.022 * min(norm, 1.25) / 1.25
            delta = vec / max(norm, 1e-8) * length
            px, py = cx + 0.235 * xx, cy + 0.220 * yy
            ax.add_patch(FancyArrowPatch(
                (px - 0.42 * delta[0], py - 0.42 * delta[1]),
                (px + 0.58 * delta[0], py + 0.58 * delta[1]),
                arrowstyle="-|>", mutation_scale=5.0,
                color=color, linewidth=1.0, alpha=0.90,
                shrinkA=0, shrinkB=0, zorder=11,
            ))


def draw_module(ax: plt.Axes, *, labels: bool) -> None:
    rng = np.random.default_rng(93)
    open_wave(ax)
    centers = [(0.235, 0.785), (0.800, 0.795), (1.365, 0.780)]
    # Deliberately large mass differences across snapshots.
    clouds = [
        cloud(centers[0], 38, (0.047, 0.093), rng),
        cloud(centers[1], 112, (0.060, 0.110), rng),
        cloud(centers[2], 58, (0.051, 0.100), rng),
    ]
    for _ in range(45):
        p0 = clouds[0][rng.integers(len(clouds[0]))]
        pm = clouds[1][rng.integers(len(clouds[1]))]
        p1 = clouds[2][rng.integers(len(clouds[2]))]
        left = brownian_segment(p0, pm, 18, rng.uniform(0.010, 0.018), rng)
        right = brownian_segment(pm, p1, 18, rng.uniform(0.010, 0.018), rng)
        path = np.vstack([left, right[1:]])
        ax.plot(path[:, 0], path[:, 1], color=PATH_PURPLE,
                linewidth=0.50, alpha=rng.uniform(0.19, 0.35), zorder=1)
    for pts in clouds:
        for (x, y), r in zip(pts, rng.uniform(0.0075, 0.0115, len(pts))):
            ax.add_patch(Circle((x, y), r, facecolor=INDIGO,
                                edgecolor=WHITE, linewidth=0.27,
                                alpha=0.89, zorder=4))
    mass_panel(ax, labels=labels)
    guidance_panel(ax, labels=labels)
    if labels:
        ax.text(0.80, 1.155, "latent unbalanced Schrödinger bridge",
                ha="center", va="top", fontsize=8.5, color=INK)


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
    fig, ax = blank_ax(); draw_module(ax, labels=False)
    save(fig, args.out_dir / "latent_unbalanced_sb_module_v3")
    fig, ax = blank_ax(); draw_module(ax, labels=True)
    save(fig, args.out_dir / "latent_unbalanced_sb_module_v3_labeled")

    # Independent Canva-ready assets. Each is redrawn on the common canvas and
    # cropped in data coordinates, preserving exactly the approved styling.
    fig, ax = blank_ax(); draw_module(ax, labels=False)
    fig.set_size_inches(5.7, 2.35)
    ax.set_xlim(0.015, 1.585); ax.set_ylim(0.455, 1.105)
    save(fig, args.out_dir / "latent_unbalanced_sb_core")

    fig, ax = blank_ax(); draw_module(ax, labels=False)
    fig.set_size_inches(2.55, 1.55)
    ax.set_xlim(0.165, 0.725); ax.set_ylim(0.040, 0.395)
    save(fig, args.out_dir / "usb_mass_change_panel")

    fig, ax = blank_ax(); draw_module(ax, labels=False)
    fig.set_size_inches(2.55, 1.55)
    ax.set_xlim(0.855, 1.435); ax.set_ylim(0.040, 0.395)
    save(fig, args.out_dir / "usb_grn_drift_shaping_panel")

    fig, ax = blank_ax(); guidance_panel_proposal(ax)
    fig.set_size_inches(4.2, 2.15)
    ax.set_xlim(0.075, 1.525); ax.set_ylim(0.055, 0.765)
    save(fig, args.out_dir / "usb_grn_drift_shaping_panel_proposal")

    for kind, filename in (
        ("sb", "usb_sb_drift_vector_field"),
        ("grn", "usb_grn_guidance_vector_field"),
        ("result", "usb_grn_guided_result_vector_field"),
    ):
        fig, ax = blank_ax(); standalone_vector_field(ax, kind)
        fig.set_size_inches(2.4, 2.0)
        ax.set_xlim(0.48, 1.12); ax.set_ylim(0.285, 0.895)
        save(fig, args.out_dir / filename)


if __name__ == "__main__":
    main()
