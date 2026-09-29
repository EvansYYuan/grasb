#!/usr/bin/env python
"""Draw the observed pancreatic time course as a shared-UMAP snapshot strip.

This is the data-backed replacement for the upper-left point-cloud sequence in
the Figure 1 AI reference.  UMAP is fitted once to all cells; every snapshot
therefore uses the same coordinates and axis limits.  No cells are connected
across time because the scRNA-seq experiment does not track individual cells.

The script writes editable SVG/PDF, a high-resolution PNG, and the UMAP
coordinates used to make them.  Re-running it reuses the coordinate CSV unless
--refit is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXPRESSION = ROOT / "data/pancreatic_preprocessed.csv"
DEFAULT_METADATA = ROOT / "data/pancreatic_metadata.tsv"
DEFAULT_OUTDIR = ROOT / "manuscript/figures"

# A compact, colorblind-conscious lineage grammar for the architecture figure.
# Fine-grained source annotations remain available in the coordinate CSV.
LINEAGE_MAP = {
    "prog_sox2": "SOX2+ progenitor",
    "prog_nkx61": "NKX6-1+ progenitor",
    "neurog3_early": "Endocrine progenitor",
    "neurog3_mid": "Endocrine progenitor",
    "neurog3_late": "Endocrine progenitor",
    "fev_high_isl_low": "Endocrine progenitor",
    "sc_alpha": "Alpha",
    "sc_beta": "Beta",
    "sst_hhex": "Delta",
    "sc_ec": "EC",
    "exo": "Exocrine",
    "phox2a": "PHOX2A+",
}

PALETTE = {
    "SOX2+ progenitor": "#777B80",
    "NKX6-1+ progenitor": "#3D82B8",
    "Endocrine progenitor": "#3AA69A",
    "Alpha": "#D9A12E",
    "Beta": "#7C62B3",
    "Delta": "#C76D3A",
    "EC": "#57A3CF",
    "Exocrine": "#B8BDC2",
    "PHOX2A+": "#B55E82",
}

ORDER = list(PALETTE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--timepoints", type=int, nargs="+", default=[0, 2, 3, 7])
    p.add_argument("--heldout", type=int, nargs="*", default=[3],
                   help="Snapshots drawn as hollow/ghosted points.")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.25)
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--fit-max-cells", type=int, default=30000,
                   help="Stratified cells used to fit PCA/UMAP; all cells are transformed.")
    p.add_argument("--max-points-per-time", type=int, default=4500,
                   help="Plot-only stratified cap; zero plots every cell.")
    p.add_argument("--refit", action="store_true")
    return p.parse_args()


def stratified_indices(labels: pd.Series, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Sample up to cap rows while approximately preserving label proportions."""
    if cap <= 0 or len(labels) <= cap:
        return np.arange(len(labels))
    chosen: list[np.ndarray] = []
    for _, positions in labels.groupby(labels, sort=False).indices.items():
        positions = np.asarray(positions)
        n = max(1, int(round(cap * len(positions) / len(labels))))
        chosen.append(rng.choice(positions, min(n, len(positions)), replace=False))
    out = np.concatenate(chosen)
    if len(out) > cap:
        out = rng.choice(out, cap, replace=False)
    return np.sort(out)


def load_inputs(expression_path: Path, metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"[load] expression: {expression_path}", flush=True)
    expression = pd.read_csv(expression_path, index_col=0)
    if "time" not in expression:
        raise ValueError("Expression table must contain a 'time' column")
    print(f"[load] metadata:   {metadata_path}", flush=True)
    metadata = pd.read_csv(metadata_path, sep="\t", index_col="library.barcode")
    missing = expression.index.difference(metadata.index)
    if len(missing):
        raise ValueError(f"{len(missing)} expression barcodes are absent from metadata")
    metadata = metadata.loc[expression.index]
    if not np.array_equal(expression["time"].to_numpy(), metadata["CellWeek"].to_numpy()):
        raise ValueError("CellWeek metadata does not agree with expression time labels")
    return expression, metadata


def compute_umap(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    lineage = metadata["Assigned_cluster"].map(LINEAGE_MAP).fillna("Other")
    strata = expression["time"].astype(str) + "|" + lineage
    fit_idx = stratified_indices(strata, args.fit_max_cells, rng)
    x = expression.drop(columns="time").to_numpy(dtype=np.float32, copy=False)

    # Scale genes without centering so the sparse-like log-expression structure
    # is retained, then denoise before UMAP. PCA and UMAP are each fitted once.
    print(f"[fit] PCA on {len(fit_idx):,}/{len(x):,} cells x {x.shape[1]:,} genes", flush=True)
    scaler = StandardScaler(with_mean=False)
    x_fit = scaler.fit_transform(x[fit_idx])
    pca = PCA(n_components=min(args.pca_components, x.shape[1]),
              svd_solver="randomized", random_state=args.seed)
    z_fit = pca.fit_transform(x_fit)
    print(f"[fit] UMAP(n_neighbors={args.neighbors}, min_dist={args.min_dist})", flush=True)
    reducer = umap.UMAP(
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        n_components=2,
        metric="euclidean",
        random_state=args.seed,
        transform_seed=args.seed,
        low_memory=True,
    )
    reducer.fit(z_fit)

    chunks = []
    for start in range(0, len(x), 5000):
        stop = min(start + 5000, len(x))
        z = pca.transform(scaler.transform(x[start:stop]))
        chunks.append(reducer.transform(z))
        print(f"[transform] {stop:,}/{len(x):,}", flush=True)
    xy = np.vstack(chunks)
    return pd.DataFrame({
        "cell": expression.index,
        "UMAP1": xy[:, 0],
        "UMAP2": xy[:, 1],
        "time": expression["time"].to_numpy(),
        "cluster": metadata["Assigned_cluster"].to_numpy(),
        "subcluster": metadata["Assigned_subcluster"].to_numpy(),
        "lineage": lineage.to_numpy(),
        # Preserve the study-provided embedding for provenance/comparison. It is
        # not used to position points in this UMAP figure.
        "study_tSNE1": metadata["tSNE_dim1"].to_numpy(),
        "study_tSNE2": metadata["tSNE_dim2"].to_numpy(),
    })


def draw_snapshot_strip(coords: pd.DataFrame, args: argparse.Namespace, stem: Path) -> None:
    rng = np.random.default_rng(args.seed)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    timepoints = args.timepoints
    fig, axes = plt.subplots(1, len(timepoints), figsize=(7.25, 2.15), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#FCFBF7")

    shown = coords[coords["time"].isin(timepoints)]
    xpad = 0.045 * (shown.UMAP1.max() - shown.UMAP1.min())
    ypad = 0.08 * (shown.UMAP2.max() - shown.UMAP2.min())
    limits = (shown.UMAP1.min() - xpad, shown.UMAP1.max() + xpad,
              shown.UMAP2.min() - ypad, shown.UMAP2.max() + ypad)

    for ax, timepoint in zip(axes, timepoints):
        ax.set_facecolor("#FCFBF7")
        frame = coords[coords["time"] == timepoint]
        keep = stratified_indices(frame["lineage"], args.max_points_per_time, rng)
        frame = frame.iloc[keep]
        ghost = timepoint in set(args.heldout)

        # Larger, pale populations first; rarer lineages remain visible on top.
        groups = sorted(frame.groupby("lineage"), key=lambda item: len(item[1]), reverse=True)
        for lineage, group in groups:
            color = PALETTE.get(lineage, "#999999")
            if ghost:
                ax.scatter(group.UMAP1, group.UMAP2, s=4.0, facecolors="none",
                           edgecolors=color, linewidths=0.28, alpha=0.33,
                           rasterized=False)
            else:
                ax.scatter(group.UMAP1, group.UMAP2, s=4.2, c=color,
                           edgecolors="none", alpha=0.72, rasterized=False)
        title = f"day {timepoint}"
        if ghost:
            title += "  (held out)"
        ax.set_title(title, color="#555B61", fontsize=8.2, pad=5)
        ax.set(xlim=limits[:2], ylim=limits[2:])
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    # A restrained time arrow above the snapshots, echoing the reference art.
    left = axes[0].get_position().x0
    right = axes[-1].get_position().x1
    y = axes[0].get_position().y1 + 0.055
    arrow = FancyArrowPatch((left, y), (right, y), transform=fig.transFigure,
                            arrowstyle="-|>", mutation_scale=10, linewidth=1.15,
                            color="#287984", shrinkA=0, shrinkB=0)
    fig.add_artist(arrow)
    for ax in axes:
        cx = (ax.get_position().x0 + ax.get_position().x1) / 2
        fig.add_artist(Line2D([cx], [y], marker="o", markersize=3.7,
                              markerfacecolor="#287984", markeredgewidth=0,
                              transform=fig.transFigure))

    present = set(coords[coords.time.isin(timepoints)].lineage)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor=PALETTE[k], markeredgewidth=0, label=k)
               for k in ORDER if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.25,
               handletextpad=0.35, fontsize=6.8)
    fig.subplots_adjust(left=0.012, right=0.995, top=0.77, bottom=0.24, wspace=0.04)

    for suffix, kwargs in {
        ".png": {"dpi": 600},
        ".pdf": {},
        ".svg": {},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", **kwargs)
        print(f"[save] {path}", flush=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coord_path = args.out_dir / "pancreatic_shared_umap_coordinates.csv.gz"
    expression, metadata = load_inputs(args.expression, args.metadata)
    if coord_path.exists() and not args.refit:
        print(f"[reuse] {coord_path}", flush=True)
        coords = pd.read_csv(coord_path)
        expected = set(expression.index)
        if set(coords["cell"]) != expected:
            raise ValueError("Cached coordinate cell IDs differ from current expression input; use --refit")
    else:
        coords = compute_umap(expression, metadata, args)
        coords.to_csv(coord_path, index=False, compression="gzip")
        print(f"[save] {coord_path}", flush=True)
    draw_snapshot_strip(coords, args, args.out_dir / "figure1_pancreatic_observed_snapshots")


if __name__ == "__main__":
    main()
