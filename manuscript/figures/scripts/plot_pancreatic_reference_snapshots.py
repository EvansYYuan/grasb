#!/usr/bin/env python
"""Create the Figure 1 observed-snapshot artwork from real pancreatic cells.

Unlike an analytical UMAP comparison, this graphical-abstract asset fits each
time point independently.  It preserves real within-time population geometry
while allowing every developmental snapshot to read clearly at small size.
There are no inferred cell-to-cell correspondences or trajectory lines.

Run with the isolated plotting environment:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/plot_pancreatic_reference_snapshots.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_EXPRESSION = PROJECT / "data/pancreatic_preprocessed.csv"
DEFAULT_METADATA = PROJECT / "data/pancreatic_metadata.tsv"
DEFAULT_OUTDIR = PROJECT / "manuscript/figures"

LINEAGE_MAP = {
    "prog_sox2": "progenitor",
    "prog_nkx61": "progenitor",
    "neurog3_early": "endocrine progenitor",
    "neurog3_mid": "endocrine progenitor",
    "neurog3_late": "endocrine progenitor",
    "fev_high_isl_low": "endocrine progenitor",
    "sc_alpha": "alpha",
    "sc_beta": "beta",
    "sst_hhex": "delta",
    "sc_ec": "EC",
    "exo": "exocrine",
    "phox2a": "other endocrine",
}

# Muted, colorblind-conscious colors taken toward the AI reference's grammar.
PALETTE = {
    "progenitor": "#737B7D",
    "endocrine progenitor": "#277DB7",
    "alpha": "#D99F2B",
    "beta": "#7C5AB5",
    "delta": "#C96D43",
    "EC": "#3FA89B",
    "exocrine": "#AEB5B8",
    "other endocrine": "#B45B83",
}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--timepoints", type=int, nargs=4, default=[0, 2, 3, 7])
    p.add_argument("--heldout", type=int, nargs="*", default=[3])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--neighbors", type=int, default=75,
                   help="Larger values favor connected global developmental structure.")
    p.add_argument("--min-dist", type=float, default=0.24)
    p.add_argument("--plot-cap", type=int, default=3500)
    p.add_argument("--include-exocrine", action="store_true",
                   help="Include the large off-lineage exocrine population.")
    p.add_argument("--refit", action="store_true")
    return p.parse_args()


def sample_stratified(labels: pd.Series, cap: int, seed: int) -> np.ndarray:
    if cap <= 0 or len(labels) <= cap:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    for _, idx in labels.groupby(labels, sort=False).indices.items():
        idx = np.asarray(idx)
        n = max(2, round(cap * len(idx) / len(labels)))
        selected.append(rng.choice(idx, min(n, len(idx)), replace=False))
    out = np.concatenate(selected)
    if len(out) > cap:
        out = rng.choice(out, cap, replace=False)
    return np.sort(out)


def orient_and_scale(xy: np.ndarray, lineage: np.ndarray) -> np.ndarray:
    """Rotate each UMAP so gray progenitors occupy the left side."""
    xy = xy - np.median(xy, axis=0)
    gray = lineage == "progenitor"
    nongray = ~gray
    if gray.any() and nongray.any():
        # The gray-to-colored centroid vector becomes the positive x-axis.
        direction = xy[nongray].mean(0) - xy[gray].mean(0)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            ex = direction / norm
            ey = np.array([-ex[1], ex[0]])
            xy = xy @ np.column_stack((ex, ey))
    else:
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        xy = xy @ vt.T

    # Robust equal-height normalization lets all four real clouds remain
    # legible in the compact architecture composition.
    lo = np.quantile(xy, [0.01], axis=0)[0]
    hi = np.quantile(xy, [0.99], axis=0)[0]
    center = (lo + hi) / 2
    scale = max(hi[1] - lo[1], 1e-6)
    return (xy - center) / scale


def fit_independent_umaps(expr: pd.DataFrame, meta: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    outputs = []
    genes = expr.columns.drop("time")
    for timepoint in args.timepoints:
        ids = expr.index[expr["time"] == timepoint]
        if not args.include_exocrine:
            ids = ids[meta.loc[ids, "Assigned_cluster"].ne("exo").to_numpy()]
        if timepoint == args.timepoints[0]:
            # The architecture's origin cloud represents the progenitor source
            # population. Day 0 also contains pre-existing mature endocrine
            # cells that form detached islands and obscure that visual role.
            source_types = {"prog_sox2", "prog_nkx61"}
            ids = ids[meta.loc[ids, "Assigned_cluster"].isin(source_types).to_numpy()]
        x = expr.loc[ids, genes].to_numpy(dtype=np.float32, copy=False)
        lineage = meta.loc[ids, "Assigned_cluster"].map(LINEAGE_MAP).fillna("other").to_numpy()
        print(f"[fit] day {timepoint}: {len(ids):,} cells", flush=True)
        x = StandardScaler().fit_transform(x)
        n_pc = min(args.pca_components, len(ids) - 1, len(genes))
        z = PCA(n_components=n_pc, svd_solver="randomized", random_state=args.seed).fit_transform(x)
        xy = umap.UMAP(
            n_neighbors=min(args.neighbors, len(ids) - 1),
            min_dist=args.min_dist,
            metric="euclidean",
            random_state=args.seed + timepoint,
            transform_seed=args.seed + timepoint,
            low_memory=True,
        ).fit_transform(z)
        xy = orient_and_scale(xy, lineage)
        outputs.append(pd.DataFrame({
            "cell": ids,
            "time": timepoint,
            "UMAP1": xy[:, 0],
            "UMAP2": xy[:, 1],
            "cluster": meta.loc[ids, "Assigned_cluster"].to_numpy(),
            "lineage": lineage,
            "study_tSNE1": meta.loc[ids, "tSNE_dim1"].to_numpy(),
            "study_tSNE2": meta.loc[ids, "tSNE_dim2"].to_numpy(),
        }))
    return pd.concat(outputs, ignore_index=True)


def draw_envelope(ax: plt.Axes, frame: pd.DataFrame, *, dashed: bool) -> None:
    """Draw a smooth outer density envelope without interior contours."""
    bins = 180
    xmin, xmax = frame.UMAP1.min(), frame.UMAP1.max()
    ymin, ymax = frame.UMAP2.min(), frame.UMAP2.max()
    xpad = max((xmax - xmin) * 0.09, 1e-3)
    ypad = max((ymax - ymin) * 0.09, 1e-3)
    hist, xedge, yedge = np.histogram2d(
        frame.UMAP1, frame.UMAP2, bins=bins,
        range=[[xmin - xpad, xmax + xpad], [ymin - ypad, ymax + ypad]],
    )
    density = gaussian_filter(hist, sigma=3.0)
    occupied = density > (density.max() * 0.012)
    occupied = binary_closing(occupied, structure=np.ones((5, 5)), iterations=3)
    occupied = binary_fill_holes(occupied)
    components, n_components = label(occupied)
    if n_components:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        occupied = components == sizes.argmax()
    xc = (xedge[:-1] + xedge[1:]) / 2
    yc = (yedge[:-1] + yedge[1:]) / 2
    ax.contour(
        xc, yc, occupied.T.astype(float), levels=[0.5],
        colors=["#8E979B" if dashed else "#536F76"],
        linewidths=1.15 if dashed else 0.82,
        linestyles=[(0, (2.4, 2.4))] if dashed else ["solid"],
        alpha=0.88 if dashed else 0.72,
    )


def draw_direction_arrow(ax: plt.Axes, frame: pd.DataFrame,
                         targets: list[str | tuple[str, ...]]) -> None:
    """Draw smooth annotation-driven gray -> blue -> mature trajectories."""
    gray = frame[frame["lineage"] == "progenitor"]
    blue = frame[frame["lineage"] == "endocrine progenitor"]
    if gray.empty or blue.empty:
        return
    start = gray[["UMAP1", "UMAP2"]].median().to_numpy()
    hub = blue[["UMAP1", "UMAP2"]].median().to_numpy()
    incoming = hub - start
    incoming_norm = np.linalg.norm(incoming)
    if incoming_norm < 1e-8:
        return
    incoming_unit = incoming / incoming_norm

    trunk_path = MplPath(
        [start, start + 0.42 * incoming, hub - 0.22 * incoming, hub],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(FancyArrowPatch(
        path=trunk_path, transform=ax.transData, arrowstyle="-",
        linewidth=1.22, color="#173F72", alpha=0.86,
        zorder=8, shrinkA=0, shrinkB=0,
    ))

    available = []
    for target_spec in targets:
        lineage_names = ((target_spec,) if isinstance(target_spec, str)
                         else target_spec)
        group = frame[frame["lineage"].isin(lineage_names)]
        if not group.empty:
            center = group[["UMAP1", "UMAP2"]].median().to_numpy()
            available.append(("/".join(lineage_names), center))
    available.sort(key=lambda item: item[1][1])
    for _, center in available:
        outgoing = center - hub
        outgoing_norm = np.linalg.norm(outgoing)
        if outgoing_norm < 1e-8:
            continue
        # Every branch initially continues along the trunk's incoming tangent,
        # then bends toward its annotated target. This is C1-like visually at
        # the shared hub while retaining one, non-overplotted trunk.
        handle = 0.20 * min(incoming_norm, outgoing_norm)
        c1 = hub + incoming_unit * handle
        c2 = center - 0.30 * outgoing
        path = MplPath(
            [hub, c1, c2, center],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        arrow = FancyArrowPatch(
            path=path, transform=ax.transData, arrowstyle="-|>",
            mutation_scale=7.6, linewidth=1.18, color="#173F72",
            alpha=0.86, zorder=8, shrinkA=0, shrinkB=2.5,
        )
        ax.add_patch(arrow)


def draw(coords: pd.DataFrame, args: argparse.Namespace, stem: Path,
         heldout_style: str = "question", enhanced: bool = False) -> None:
    # STIX is an open-source Times-like family bundled with Matplotlib. The
    # default artwork is text-free, but this makes future labeling consistent.
    mpl.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.62))
    fig.patch.set_alpha(0)
    heldout = set(args.heldout)

    # Deliberately uneven relative footprint, echoing the reference: compact
    # origin, increasingly structured later distributions, ghost in between.
    visual_scale = {args.timepoints[0]: 0.76, args.timepoints[1]: 0.94,
                    args.timepoints[2]: 0.91, args.timepoints[3]: 1.00}
    for panel, (ax, timepoint) in enumerate(zip(axes, args.timepoints)):
        ax.set_facecolor("none")
        frame = coords[coords["time"] == timepoint].copy()
        keep = sample_stratified(frame["lineage"], args.plot_cap, args.seed + timepoint)
        frame = frame.iloc[keep]
        scale = visual_scale[timepoint]
        frame[["UMAP1", "UMAP2"]] *= scale

        if timepoint in heldout:
            if heldout_style == "question":
                # A schematic unknown snapshot cannot be confused with gray
                # progenitor cells. Coordinates remain cached for analysis.
                circle = Circle((0.5, 0.5), 0.205, transform=ax.transAxes,
                                facecolor="none", edgecolor="#929A9E",
                                linewidth=1.15, linestyle=(0, (2.2, 2.2)))
                ax.add_patch(circle)
                ax.text(0.5, 0.505, "?", transform=ax.transAxes,
                        ha="center", va="center", color="#737C80",
                        fontsize=24, fontweight="normal")
            elif heldout_style == "hollow_shape":
                # Trace a smoothed density envelope around the real UMAP. The
                # interior contains no cell marks and therefore cannot read as
                # another gray cell population.
                draw_envelope(ax, frame, dashed=True)
            else:
                raise ValueError(f"Unknown heldout style: {heldout_style}")
        elif panel == 0:
            ax.scatter(frame.UMAP1, frame.UMAP2, s=2.4, c="#777D80",
                       edgecolors="none", alpha=0.66)
        else:
            groups = sorted(frame.groupby("lineage"), key=lambda item: len(item[1]), reverse=True)
            for lineage, group in groups:
                ax.scatter(group.UMAP1, group.UMAP2, s=2.4,
                           c=PALETTE.get(lineage, "#92999C"),
                           edgecolors="none", alpha=0.72)
        if enhanced and timepoint not in heldout:
            draw_envelope(ax, frame, dashed=False)
            if panel in (1, 3):
                targets = ([("alpha", "beta"), "other endocrine"] if panel == 1 else
                           ["alpha", "beta", "delta", "EC", "other endocrine"])
                draw_direction_arrow(ax, frame, targets=targets)
        xmin, xmax = frame.UMAP1.min(), frame.UMAP1.max()
        ymin, ymax = frame.UMAP2.min(), frame.UMAP2.max()
        xpad = max((xmax - xmin) * 0.13, 0.05)
        ypad = max((ymax - ymin) * 0.13, 0.05)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # Reference-style time rail: dots correspond to the four snapshots; the
    # held-out position is hollow. No labels are needed in the architecture.
    fig.subplots_adjust(left=0.006, right=0.994, top=0.79, bottom=0.015, wspace=0.005)
    centers = [(ax.get_position().x0 + ax.get_position().x1) / 2 for ax in axes]
    rail_y = 0.895
    rail_start = centers[0] - 0.075
    rail_end = centers[-1] + 0.085
    heldout_center = centers[args.timepoints.index(next(iter(heldout)))]
    # 4.3 pt marker diameter on a 7.2 inch canvas -> radius ~0.00415 in
    # figure coordinates. Segments terminate exactly at that circumference.
    marker_gap = 0.00415
    fig.add_artist(Line2D(
        [rail_start, heldout_center - marker_gap], [rail_y, rail_y],
        transform=fig.transFigure, color="#287B86", linewidth=1.25,
        solid_capstyle="butt",
    ))
    rail = FancyArrowPatch((heldout_center + marker_gap, rail_y),
                           (rail_end, rail_y), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=10.5,
                           linewidth=1.25, color="#287B86",
                           shrinkA=0, shrinkB=0)
    fig.add_artist(rail)
    for center, timepoint in zip(centers, args.timepoints):
        hollow = timepoint in heldout
        fig.add_artist(Line2D(
            [center], [rail_y], transform=fig.transFigure,
            marker="o", linestyle="none", markersize=4.3,
            markerfacecolor="none" if hollow else "#287B86",
            markeredgecolor="#287B86", markeredgewidth=1.0,
        ))
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.015, **kwargs)
        print(f"[save] {path}", flush=True)
    plt.close(fig)


def draw_standalone(coords: pd.DataFrame, args: argparse.Namespace, timepoint: int,
                    stem: Path, enhanced: bool = False) -> None:
    """Export one large, text-free snapshot using the same visual grammar."""
    frame = coords[coords["time"] == timepoint].copy()
    keep = sample_stratified(frame["lineage"], args.plot_cap, args.seed + timepoint)
    frame = frame.iloc[keep]
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    groups = sorted(frame.groupby("lineage"), key=lambda item: len(item[1]), reverse=True)
    for lineage, group in groups:
        ax.scatter(group.UMAP1, group.UMAP2, s=4.2,
                   c=PALETTE.get(lineage, "#92999C"), edgecolors="none", alpha=0.74)
    if enhanced:
        draw_envelope(ax, frame, dashed=False)
        if timepoint in (args.timepoints[1], args.timepoints[3]):
            targets = ([("alpha", "beta"), "other endocrine"] if timepoint == args.timepoints[1]
                       else ["alpha", "beta", "delta", "EC", "other endocrine"])
            draw_direction_arrow(ax, frame, targets=targets)
    xmin, xmax = frame.UMAP1.min(), frame.UMAP1.max()
    ymin, ymax = frame.UMAP2.min(), frame.UMAP2.max()
    xpad = max((xmax - xmin) * 0.13, 0.05)
    ypad = max((ymax - ymin) * 0.13, 0.05)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.015, **kwargs)
        print(f"[save] {path}", flush=True)
    plt.close(fig)


def main() -> None:
    args = arguments()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scope = "all_cells" if args.include_exocrine else "endocrine_focus"
    cache = args.out_dir / f"pancreatic_independent_umap_{scope}_coordinates.csv.gz"
    expr = pd.read_csv(args.expression, index_col=0)
    meta = pd.read_csv(args.metadata, sep="\t", index_col="library.barcode").loc[expr.index]
    if cache.exists() and not args.refit:
        coords = pd.read_csv(cache)
        if sorted(coords.time.unique()) != sorted(args.timepoints):
            raise ValueError("Cached time points differ; rerun with --refit")
        print(f"[reuse] {cache}", flush=True)
    else:
        coords = fit_independent_umaps(expr, meta, args)
        coords.to_csv(cache, index=False, compression="gzip")
        print(f"[save] {cache}", flush=True)
    draw(coords, args,
         args.out_dir / "figure1_pancreatic_reference_snapshots_question_placeholder",
         heldout_style="question")
    draw(coords, args,
         args.out_dir / "figure1_pancreatic_reference_snapshots_hollow_heldout",
         heldout_style="hollow_shape")
    draw(coords, args,
         args.out_dir / "figure1_pancreatic_reference_snapshots_trajectory_outlines",
         heldout_style="hollow_shape", enhanced=True)
    for ordinal in (0, 1, 3):
        timepoint = args.timepoints[ordinal]
        draw_standalone(
            coords, args, timepoint,
            args.out_dir / f"figure1_pancreatic_snapshot_{ordinal + 1}_day{timepoint}",
        )
        draw_standalone(
            coords, args, timepoint,
            args.out_dir / f"figure1_pancreatic_snapshot_{ordinal + 1}_day{timepoint}_outlined_arrow",
            enhanced=True,
        )


if __name__ == "__main__":
    main()
