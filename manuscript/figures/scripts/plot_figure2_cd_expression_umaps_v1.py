#!/usr/bin/env python
"""Render revised Figure 2c-d as observed-expression UMAP feature plots.

Panel c shows the six pancreatic genes annotated in Figure 2a. Panel d shows
the eight zebrafish genes annotated in Figure 2b, including MYF5. Coordinates
are fitted from the observed expression matrices only; model outputs do not
enter either embedding or feature color.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

import plot_drift_expression_context_v1 as style


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
FIGURE_DIR = REPO / "manuscript/figures"
DATA_DIR = REPO / "manuscript/figures/data"

PANCREAS_EXPR = DATA / "pancreatic_preprocessed.csv"
PANCREAS_META = DATA / "pancreatic_metadata.tsv"
PANCREAS_COORDS = {
    3.0: DATA_DIR / "pancreas_day3_endocrine_umap_n250_md035_coordinates.csv.gz",
    6.0: DATA_DIR / "pancreas_day6_endocrine_umap_n800_md035_coordinates.csv.gz",
}
ZEBRAFISH_EXPR = DATA / "zebrafish_preprocessed.csv"
ZEBRAFISH_META = DATA / "zebrafish_metadata.csv"
ZEBRAFISH_GROUPS = DATA / "zebrafish_groups.csv"
ZEBRAFISH_SEGMENTS = REPO / "config/zebrafish/segment_map.csv"
ZEBRAFISH_COORDS = DATA_DIR / "zebrafish_9hpf_mesoderm_umap_n300_md035_coordinates.csv.gz"

OUTPUT_C_STEM = FIGURE_DIR / "figure2c_expression_umaps"
OUTPUT_D_STEM = FIGURE_DIR / "figure2d_expression_umaps"
SEED = 17
HPF = 9.0
UMAP_NEIGHBORS = {3.0: 250, 6.0: 800}
UMAP_MIN_DIST = 0.35

PANCREAS_REFERENCE_LINEAGES = {
    "sc_beta", "sc_alpha", "sc_ec", "sst_hhex", "neurog3_early",
    "neurog3_mid", "neurog3_late", "prog_nkx61", "phox2a",
    "fev_high_isl_low",
}
ZEBRAFISH_MESODERM_LINEAGES = {
    "somites", "adaxial_cells", "notochord", "prechordal_plate",
}

PANCREAS_SPECS = [
    ("NEUROG3", "endocrine progenitors", 3.0,
     ("neurog3_early", "neurog3_mid", "neurog3_late")),
    ("FEV", "endocrine progenitor " r"$\rightarrow$" " EC-like\ntransition intermediates", 3.0,
     ("neurog3_late", "fev_high_isl_low", "phox2a", "sc_ec")),
    ("SST", "delta-like", 6.0, "sst_hhex"),
    ("PAX4", "endocrine progenitors", 3.0,
     ("neurog3_early", "neurog3_mid", "neurog3_late")),
    ("DDC", "endocrine progenitor " r"$\rightarrow$" " EC-like\ntransition intermediates", 3.0,
     ("neurog3_late", "fev_high_isl_low", "phox2a", "sc_ec")),
    ("ARX", "alpha-like", 6.0, "sc_alpha"),
]

ZEBRAFISH_SPECS = [
    ("RIPPLY1", "somite / prechordal-plate cells", 9.0,
     ("somites", "prechordal_plate")),
    ("MYF5", "somitic and adaxial myogenic cells", 9.0,
     ("somites", "adaxial_cells")),
    ("CDKN1CA", "adaxial and notochord cells", 9.0,
     ("adaxial_cells", "notochord")),
    ("TA", "notochord", 9.0, "notochord"),
    ("PCDH8", "somitic and adaxial cells", 9.0,
     ("somites", "adaxial_cells")),
    ("MYOD1", "adaxial slow-muscle precursors", 9.0, "adaxial_cells"),
    ("TBX16", "somitic and adaxial cells", 9.0,
     ("somites", "adaxial_cells")),
    ("PMP22B", "notochord", 9.0, "notochord"),
]

EXPRESSION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grn_expression", ["#F5D9B5", "#E99555", "#D85249", "#7E2032"]
)
EXPRESSION_NORM = mcolors.PowerNorm(gamma=1.25, vmin=0, vmax=1)


def load_selected_expression(path: Path, genes: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    first = header[0]
    missing = sorted(set(genes).difference(header))
    if missing:
        raise ValueError(f"Missing genes in {path}: {missing}")
    return pd.read_csv(path, usecols=[first, *genes], index_col=0)


def load_expression_rows(path: Path, cells: pd.Index) -> pd.DataFrame:
    wanted = set(cells.astype(str))
    chunks = []
    for chunk in pd.read_csv(path, index_col=0, chunksize=4000):
        use = chunk.loc[chunk.index.astype(str).isin(wanted)]
        if not use.empty:
            chunks.append(use)
    frame = pd.concat(chunks)
    missing = cells.difference(frame.index)
    if len(missing):
        raise ValueError(f"{len(missing):,} requested cells are absent from {path}")
    return frame.loc[cells]


def fit_single_stage_umap(frame: pd.DataFrame, seed: int, n_neighbors: int) -> np.ndarray:
    """Fit a UMAP using only cells from one observed stage."""
    genes = frame.columns.drop("time")
    x = frame[genes].to_numpy(dtype=np.float32, copy=False)
    # The matrices are already normalized/log-transformed and restricted to
    # the study's HVGs. Avoid a second variance scaling, which overemphasizes
    # sparse low-variance genes. Remove only invariant genes before PCA.
    variance = np.var(x, axis=0)
    x = x[:, variance > 1e-8]
    n_pc = min(40, x.shape[0] - 1, x.shape[1])
    z = PCA(n_components=n_pc, svd_solver="randomized", random_state=seed).fit_transform(x)
    return umap.UMAP(
        n_neighbors=min(n_neighbors, len(frame) - 1),
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        init="pca",
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
        n_jobs=1,
    ).fit_transform(z)


def fit_pancreas_umaps() -> pd.DataFrame:
    meta = pd.read_csv(PANCREAS_META, sep="\t", index_col="library.barcode")
    outputs = []
    for stage, path in PANCREAS_COORDS.items():
        cells = meta.index[
            pd.to_numeric(meta.CellWeek, errors="coerce").eq(stage)
            & meta.Assigned_cluster.isin(PANCREAS_REFERENCE_LINEAGES)
        ]
        print(f"[load] pancreatic day {stage:g}: {len(cells):,} cells", flush=True)
        frame = load_expression_rows(PANCREAS_EXPR, cells)
        xy = fit_single_stage_umap(frame, SEED + int(stage), UMAP_NEIGHBORS[stage])
        coords = pd.DataFrame({
            "cell": cells,
            "UMAP1": xy[:, 0],
            "UMAP2": xy[:, 1],
            "time": stage,
            "lineage": meta.loc[cells, "Assigned_cluster"].to_numpy(),
        })
        coords.to_csv(path, index=False, compression="gzip")
        print(f"[write] {path}", flush=True)
        outputs.append(coords)
    return pd.concat(outputs, ignore_index=True).set_index("cell")


def load_pancreas() -> tuple[pd.DataFrame, pd.DataFrame]:
    if all(path.exists() for path in PANCREAS_COORDS.values()):
        coords = pd.concat([pd.read_csv(path) for path in PANCREAS_COORDS.values()],
                           ignore_index=True).set_index("cell")
    else:
        coords = fit_pancreas_umaps()
    genes = [g for g, _, _, _ in PANCREAS_SPECS]
    expr = load_selected_expression(PANCREAS_EXPR, genes)
    common = coords.index.intersection(expr.index, sort=False)
    if len(common) != len(coords):
        raise ValueError(f"Only {len(common):,}/{len(coords):,} pancreatic UMAP cells have expression")
    return coords.loc[common], expr.loc[common]


def fit_zebrafish_umap() -> pd.DataFrame:
    """Fit one stage-matched UMAP to all observed 9-hpf cells."""
    print(f"[load] observed zebrafish expression at {HPF:g} hpf", flush=True)
    meta_all = pd.read_csv(ZEBRAFISH_META, index_col=0)
    groups_all = pd.read_csv(ZEBRAFISH_GROUPS, index_col=0).reindex(meta_all.index)
    segment_map = pd.read_csv(ZEBRAFISH_SEGMENTS)
    to_lineage = dict(zip(segment_map.segment.astype(float), segment_map.lineage.astype(str)))
    lineage_all = pd.to_numeric(groups_all.segment, errors="coerce").map(to_lineage)
    cells = meta_all.index[
        pd.to_numeric(meta_all.HPF, errors="coerce").eq(HPF)
        & lineage_all.isin(ZEBRAFISH_MESODERM_LINEAGES)
    ]
    frame = load_expression_rows(ZEBRAFISH_EXPR, cells)
    print(f"[fit] single-stage PCA/UMAP: {len(frame):,} cells", flush=True)
    xy = fit_single_stage_umap(frame, SEED + int(HPF), 300)

    meta = meta_all.reindex(frame.index)
    groups = groups_all.reindex(frame.index)
    lineage = pd.to_numeric(groups.segment, errors="coerce").map(to_lineage).fillna("other")
    coords = pd.DataFrame({
        "cell": frame.index,
        "UMAP1": xy[:, 0],
        "UMAP2": xy[:, 1],
        "HPF": meta.HPF.to_numpy(),
        "segment": pd.to_numeric(groups.segment, errors="coerce").to_numpy(),
        "lineage": lineage.to_numpy(),
    })
    coords.to_csv(ZEBRAFISH_COORDS, index=False, compression="gzip")
    print(f"[write] {ZEBRAFISH_COORDS}", flush=True)
    return coords.set_index("cell")


def load_zebrafish() -> tuple[pd.DataFrame, pd.DataFrame]:
    if ZEBRAFISH_COORDS.exists():
        coords = pd.read_csv(ZEBRAFISH_COORDS).set_index("cell")
    else:
        coords = fit_zebrafish_umap()
    genes = [g for g, _, _, _ in ZEBRAFISH_SPECS]
    expr = load_selected_expression(ZEBRAFISH_EXPR, genes)
    common = coords.index.intersection(expr.index, sort=False)
    if len(common) != len(coords):
        raise ValueError(f"Only {len(common):,}/{len(coords):,} zebrafish UMAP cells have expression")
    return coords.loc[common], expr.loc[common]


def limits(coords: pd.DataFrame) -> tuple[float, float, float, float]:
    x0, x1 = coords.UMAP1.quantile([0.002, 0.998])
    y0, y1 = coords.UMAP2.quantile([0.002, 0.998])
    padx = 0.035 * (x1 - x0)
    pady = 0.035 * (y1 - y0)
    return x0 - padx, x1 + padx, y0 - pady, y1 + pady


def feature_plot(
    ax: plt.Axes,
    coords: pd.DataFrame,
    expression: pd.Series,
    gene: str,
    cell_type: str,
    target_lineage: str | tuple[str, ...],
    stage_label: str,
    xy_limits: tuple[float, float, float, float],
    point_size: float,
) -> None:
    values = pd.to_numeric(expression, errors="coerce").fillna(0).clip(lower=0)
    ax.scatter(coords.UMAP1, coords.UMAP2, s=point_size, color="#CDD2D4",
               alpha=0.48, edgecolors="none", rasterized=True, zorder=1)
    target_lineages = (target_lineage,) if isinstance(target_lineage, str) else target_lineage
    target = coords.lineage.astype(str).isin(target_lineages)
    if target.any():
        ax.scatter(coords.loc[target, "UMAP1"], coords.loc[target, "UMAP2"],
                   s=point_size * 5.2, color="#368F86", alpha=0.92,
                   edgecolors="none", rasterized=True, zorder=2)
    positive = values.gt(0)
    if positive.any():
        vmax = float(values[positive].quantile(0.99))
        vmax = max(vmax, float(values[positive].max()) * 0.05, 1e-8)
        scaled = (values[positive] / vmax).clip(0, 1)
        order = np.argsort(scaled.to_numpy())
        cells = scaled.index[order]
        rgba = EXPRESSION_CMAP(EXPRESSION_NORM(scaled.loc[cells].to_numpy()))
        rgba[:, 3] = 0.94
        ax.scatter(coords.loc[cells, "UMAP1"], coords.loc[cells, "UMAP2"],
                   s=point_size * 1.8, c=rgba,
                   edgecolors="none",
                   rasterized=True, zorder=3)
    ax.set(xlim=xy_limits[:2], ylim=xy_limits[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    display_gene = gene.replace("-", r"\!-\!")
    ax.set_title(rf"$\it{{{display_gene}}}$" + f"\n{cell_type}\n{stage_label}", fontsize=5.15,
                 color=style.OBSERVED, pad=1.5, linespacing=1.02)


def plot_species(
    fig: plt.Figure,
    slot,
    coords: pd.DataFrame,
    expr: pd.DataFrame,
    specs: list[tuple[str, str, float, str]],
    shape: tuple[int, int],
    letter: str,
    dataset_label: str,
    stage_column: str,
    stage_unit: str,
    point_size: float,
) -> None:
    host = fig.add_subplot(slot)
    host.axis("off")
    host.text(-0.025, 1.02, letter, transform=host.transAxes, fontsize=11,
              fontweight="bold", ha="left", va="bottom", clip_on=False)
    host.text(0.04, 1.02, dataset_label, transform=host.transAxes, fontsize=7.0,
              fontweight="bold", color="#495156", ha="left", va="bottom")
    grid = GridSpecFromSubplotSpec(shape[0] + 1, shape[1], subplot_spec=slot,
                                   height_ratios=[0.16, *([1.0] * shape[0])],
                                   hspace=0.42, wspace=0.08)
    stages = sorted(set(stage for _, _, stage, _ in specs))
    stage_coords = {
        stage: coords[pd.to_numeric(coords[stage_column], errors="coerce").eq(stage)]
        for stage in stages
    }
    stage_limits = {stage: limits(frame) for stage, frame in stage_coords.items()}
    for idx, (gene, cell_type, stage, target_lineage) in enumerate(specs):
        ax = fig.add_subplot(grid[1 + idx // shape[1], idx % shape[1]])
        frame = stage_coords[stage]
        stage_label = f"day {int(stage)}" if stage_unit == "day" else f"{stage:g} hpf"
        feature_plot(ax, frame, expr.loc[frame.index, gene], gene, cell_type, target_lineage,
                     stage_label, stage_limits[stage], point_size)


def add_legend_and_colorbar(fig: plt.Figure) -> None:
    legend = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="#CDD2D4", markeredgewidth=0, label="other cells"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.2,
               markerfacecolor="#368F86", markeredgewidth=0,
               label="selected cell population"),
    ]
    fig.legend(handles=legend, loc="lower left", ncol=2, frameon=False,
               bbox_to_anchor=(0.19, 0.035), fontsize=6.0,
               columnspacing=1.1, handletextpad=0.35)
    cax = fig.add_axes([0.64, 0.065, 0.24, 0.018])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=EXPRESSION_NORM,
                                                  cmap=EXPRESSION_CMAP),
                            cax=cax, orientation="horizontal")
    colorbar.set_ticks([0, 0.5, 1], labels=["low", "", "high"])
    colorbar.ax.tick_params(labelsize=5.3, length=0, pad=1.2)
    colorbar.outline.set_linewidth(0.35)
    colorbar.set_label("expression level of selected gene", fontsize=5.8, labelpad=1.8)


def main() -> None:
    style.register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p_coords, p_expr = load_pancreas()
    z_coords, z_expr = load_zebrafish()

    fig_c = plt.figure(figsize=(style.WIDTH_IN, 4.45))
    outer_c = GridSpec(1, 1, figure=fig_c, left=0.045, right=0.985, top=0.92, bottom=0.19)
    plot_species(fig_c, outer_c[0], p_coords, p_expr, PANCREAS_SPECS, (2, 3),
                 "c", "Pancreas", "time", "day", point_size=0.82)
    add_legend_and_colorbar(fig_c)
    style.save(fig_c, OUTPUT_C_STEM)

    fig_d = plt.figure(figsize=(style.WIDTH_IN, 4.15))
    outer_d = GridSpec(1, 1, figure=fig_d, left=0.045, right=0.985, top=0.92, bottom=0.20)
    plot_species(fig_d, outer_d[0], z_coords, z_expr, ZEBRAFISH_SPECS, (2, 4),
                 "d", "Zebrafish", "HPF", "hpf", point_size=1.25)
    add_legend_and_colorbar(fig_d)
    style.save(fig_d, OUTPUT_D_STEM)

    manifest = {
        "version": "Figure 2c-d observed-expression UMAP review",
        "argument": "annotated rank-shift genes occupy cell-type-specific regions in observed expression space",
        "pancreas_embeddings": [str(path.relative_to(REPO))
                                for path in PANCREAS_COORDS.values()],
        "pancreas_expression": str(PANCREAS_EXPR),
        "zebrafish_expression": str(ZEBRAFISH_EXPR),
        "zebrafish_embedding": str(ZEBRAFISH_COORDS.relative_to(REPO)),
        "expression_encoding": (
            "zero values retain the grey/teal base layer; positive values are scaled to each "
            "gene's 99th percentile and shown with a continuous light-orange-to-burgundy gradient"
        ),
        "stage_logic": "each feature map is restricted to the stage used for that gene's rank point",
        "embedding_scope": (
            "independent PCA-UMAP fits for pancreatic day 3, pancreatic day 6, and "
            "zebrafish 9 hpf; no pooled-stage embedding"
        ),
        "cell_universe": (
            "pancreas: endocrine-lineage annotation set at the selected day; zebrafish: "
            "somites, adaxial cells, notochord, and prechordal plate at 9 hpf; no filtering "
            "by focal-gene expression"
        ),
        "umap_parameters": {"n_neighbors_by_day": UMAP_NEIGHBORS,
                            "min_dist": UMAP_MIN_DIST},
        "pancreas_genes": [{"gene": g, "cell_type": ct, "day": t, "lineage": lin}
                           for g, ct, t, lin in PANCREAS_SPECS],
        "zebrafish_genes": [{"gene": g, "cell_type": ct, "hpf": t, "lineage": lin}
                            for g, ct, t, lin in ZEBRAFISH_SPECS],
        "outputs": [f"{stem.name}.{ext}" for stem in (OUTPUT_C_STEM, OUTPUT_D_STEM)
                    for ext in ("pdf", "svg", "png")],
    }
    (DATA_DIR / "figure2_cd_expression_umaps_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
