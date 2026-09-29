#!/usr/bin/env python
"""Prototype Figure 3: expression violins aligned to pancreatic drift ranks.

For each held-out lineage--day example, the upper row shows observed expression
for a fixed gene panel. The lower row places no-GRN and genuine-GRN decoded-
drift ranks at the same gene positions. Genes are ordered by observed mean
expression so any association between abundance and ranking is visible rather
than asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

import plot_drift_expression_context_v1 as style


REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO / "manuscript/figures"
DATA_DIR = REPO / "manuscript/figures/data"
DRIFT = REPO / "results/per_type_drift.csv"
EXPRESSION = REPO / "data/pancreatic_preprocessed.csv"
METADATA = REPO / "data/pancreatic_metadata.tsv"
OUTPUT_STEM = FIGURE_DIR / "figure3_pancreas_ranked_violins_v1"
DATA_OUTPUT = DATA_DIR / "figure3_pancreas_ranked_violins_v1.csv"
MANIFEST_OUTPUT = DATA_DIR / "figure3_pancreas_ranked_violins_v1_manifest.json"

UNITS = [
    {
        "stage": 3,
        "cell_type": "neurog3_late",
        "title": "Day 3 · late endocrine progenitors",
        "panel": "a",
        "genes": ["INS", "CALB2", "MAFB", "NEUROG3", "NEUROD1", "FEV", "DDC", "PAX4"],
    },
    {
        "stage": 3,
        "cell_type": "sc_alpha",
        "title": "Day 3 · alpha-like cells",
        "panel": "b",
        "genes": ["GCG", "INS", "SCG5", "SLC30A8", "CDKN1C", "MAFB", "RGS4", "ARX"],
    },
    {
        "stage": 6,
        "cell_type": "sc_beta",
        "title": "Day 6 · beta-like cells",
        "panel": "c",
        "genes": ["INS", "CALB2", "ACVR1C", "SCG5", "SLC30A8", "ERO1B", "IGFBP5", "ISL1"],
    },
    {
        "stage": 6,
        "cell_type": "sc_ec",
        "title": "Day 6 · EC-like cells",
        "panel": "d",
        "genes": ["INS", "CALB2", "GAL", "SCG2", "CDKN1C", "TPH1", "GCH1", "IGFBP5"],
    },
]
CONDITIONS = [
    ("latent_nogrn", "No GRN", style.NO_GRN),
    ("latent_grn", "Genuine GRN", style.GRN),
]


def load_expression(genes: list[str], cells: pd.Index) -> pd.DataFrame:
    columns = pd.read_csv(EXPRESSION, nrows=0).columns.tolist()
    index_column = columns[0]
    missing = sorted(set(genes).difference(columns))
    if missing:
        raise ValueError(f"Genes absent from expression matrix: {missing}")
    frame = pd.read_csv(EXPRESSION, usecols=[index_column, *genes], index_col=0)
    missing_cells = cells.difference(frame.index)
    if len(missing_cells):
        raise ValueError(f"{len(missing_cells):,} selected cells are absent from expression matrix")
    return frame.loc[cells]


def prepare() -> tuple[pd.DataFrame, dict[tuple[int, str], pd.DataFrame]]:
    drift = pd.read_csv(DRIFT)
    metadata = pd.read_csv(METADATA, sep="\t", index_col="library.barcode")
    all_genes = sorted({gene for unit in UNITS for gene in unit["genes"]})
    cells_by_unit: dict[tuple[int, str], pd.Index] = {}
    all_cells = pd.Index([])
    for unit in UNITS:
        key = (unit["stage"], unit["cell_type"])
        cells = metadata.index[
            pd.to_numeric(metadata.CellWeek, errors="coerce").eq(unit["stage"])
            & metadata.Assigned_cluster.eq(unit["cell_type"])
        ]
        if len(cells) < 2:
            raise ValueError(f"Too few cells for {key}: {len(cells)}")
        cells_by_unit[key] = cells
        all_cells = all_cells.union(cells)

    expression = load_expression(all_genes, all_cells)
    expression_by_unit = {key: expression.loc[cells] for key, cells in cells_by_unit.items()}
    rows = []
    for unit in UNITS:
        key = (unit["stage"], unit["cell_type"])
        observed = expression_by_unit[key]
        condition_tables = {}
        for tag, label, _ in CONDITIONS:
            q = drift[
                drift.tag.eq(tag)
                & drift.stage.eq(unit["stage"])
                & drift.cell_type.eq(unit["cell_type"])
            ].sort_values("drift", ascending=False).reset_index(drop=True)
            q["rank"] = np.arange(1, len(q) + 1)
            condition_tables[label] = q.set_index("gene")
        for gene in unit["genes"]:
            values = observed[gene]
            row = {
                "stage": unit["stage"],
                "cell_type": unit["cell_type"],
                "gene": gene,
                "n_cells": len(values),
                "expression_mean": values.mean(),
                "expression_median": values.median(),
                "expression_q25": values.quantile(0.25),
                "expression_q75": values.quantile(0.75),
                "detection_fraction": values.gt(0).mean(),
            }
            for label, table in condition_tables.items():
                prefix = "nogrn" if label == "No GRN" else "grn"
                row[f"{prefix}_rank"] = int(table.loc[gene, "rank"])
                row[f"{prefix}_drift"] = float(table.loc[gene, "drift"])
            rows.append(row)
    return pd.DataFrame(rows), expression_by_unit


def expression_violins(
    ax: plt.Axes,
    ordered: pd.DataFrame,
    expression: pd.DataFrame,
    show_ylabel: bool,
) -> None:
    genes = ordered.gene.tolist()
    values = [expression[gene].to_numpy() for gene in genes]
    positions = np.arange(1, len(genes) + 1)
    parts = ax.violinplot(values, positions=positions, widths=0.72,
                          showmeans=False, showmedians=False, showextrema=False,
                          bw_method=0.25)
    for body in parts["bodies"]:
        body.set_facecolor(style.TEAL)
        body.set_edgecolor(style.TEAL)
        body.set_linewidth(0.55)
        body.set_alpha(0.28)
    ax.set_xlim(0.45, len(genes) + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Observed normalized\nexpression" if show_ylabel else "")
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_visible(True)


def aligned_ranks(ax: plt.Axes, ordered: pd.DataFrame) -> None:
    positions = np.arange(1, len(ordered) + 1)
    no_rank = ordered.nogrn_rank.to_numpy()
    grn_rank = ordered.grn_rank.to_numpy()
    for x, r0, r1 in zip(positions, no_rank, grn_rank):
        ax.plot([x, x], [r0, r1], color="#AEB5B8", linewidth=0.7, zorder=1)
    ax.scatter(positions, no_rank, s=22, facecolor="white", edgecolor=style.NO_GRN,
               linewidth=0.85, zorder=3, clip_on=False)
    ax.scatter(positions, grn_rank, s=22, facecolor=style.GRN, edgecolor=style.GRN,
               linewidth=0.75, zorder=4, clip_on=False)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(3500, 0.45)
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["1", "10", "100", "1000"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.45, len(ordered) + 0.55)
    ax.set_xticks(positions)
    ax.set_xticklabels([rf"$\it{{{gene}}}$" for gene in ordered.gene],
                       rotation=48, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Decoded-drift rank\n(1 is best)")
    ax.grid(axis="y", color="#E8EBEC", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    style.register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary, expression_by_unit = prepare()
    ordered_units = []
    for unit in UNITS:
        q = summary[
            summary.stage.eq(unit["stage"])
            & summary.cell_type.eq(unit["cell_type"])
        ].sort_values(["expression_mean", "gene"], ascending=[False, True]).copy()
        q["expression_order"] = np.arange(1, len(q) + 1)
        ordered_units.append(q)
    rendered = pd.concat(ordered_units, ignore_index=True)
    rendered.to_csv(DATA_OUTPUT, index=False)

    fig = plt.figure(figsize=(style.WIDTH_IN, 7.35))
    outer = GridSpec(2, 2, figure=fig, left=0.09, right=0.985, bottom=0.145,
                     top=0.95, wspace=0.34, hspace=0.31)
    expression_axes = []
    for panel_index, unit in enumerate(UNITS):
        row, column = divmod(panel_index, 2)
        ordered = rendered[
            rendered.stage.eq(unit["stage"])
            & rendered.cell_type.eq(unit["cell_type"])
        ].sort_values("expression_order")
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, column],
                                        height_ratios=[1.0, 0.92], hspace=0.28)
        expr_ax = fig.add_subplot(inner[0])
        rank_ax = fig.add_subplot(inner[1])
        expression_violins(
            expr_ax,
            ordered,
            expression_by_unit[(unit["stage"], unit["cell_type"])],
            show_ylabel=True,
        )
        aligned_ranks(rank_ax, ordered)
        expr_ax.set_title(unit["title"], fontsize=8.2, fontweight="bold", pad=8)
        expr_ax.text(-0.18, 1.18, unit["panel"], transform=expr_ax.transAxes,
                     fontsize=10.5, fontweight="bold", ha="left", va="top",
                     clip_on=False)
        expression_axes.append(expr_ax)

    expression_ymax = max(ax.get_ylim()[1] for ax in expression_axes)
    for ax in expression_axes:
        ax.set_ylim(-0.12, expression_ymax)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor="white", markeredgecolor=style.NO_GRN,
               markeredgewidth=0.85, label="No GRN"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=style.GRN, markeredgecolor=style.GRN,
               label="Genuine GRN"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.025), handletextpad=0.4, columnspacing=1.5)

    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.015}
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), **save_kwargs)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=style.PNG_DPI, **save_kwargs)
    plt.close(fig)
    manifest = {
        "version": "Figure 3 pancreatic expression-aligned ranks v1",
        "argument": (
            "Observed expression and model-specific decoded-drift ranks are aligned for a fixed "
            "gene panel, exposing when no-GRN prioritization follows abundant state markers and "
            "when GRN guidance elevates regulatory genes."
        ),
        "units": UNITS,
        "gene_order": "descending observed mean expression within the matching lineage and day",
        "rank_definition": "descending positive decoded drift within model, lineage, and day",
        "expression_definition": (
            "observed normalized expression across cells in the matching annotated lineage and day"
        ),
        "sources": [
            str(DRIFT.relative_to(REPO)),
            str(EXPRESSION.relative_to(REPO)),
            str(METADATA.relative_to(REPO)),
        ],
        "outputs": [f"{OUTPUT_STEM.name}.{ext}" for ext in ("pdf", "svg", "png")]
                   + [DATA_OUTPUT.name],
    }
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
