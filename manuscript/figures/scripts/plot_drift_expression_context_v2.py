#!/usr/bin/env python
"""Revise Figure 2 selection panels for direct biological interpretation.

Panels 2a-b place prespecified reference genes on paired no-GRN and genuine-GRN
decoded-drift ranks.  The selected exemplars are annotated by their biological role.
The manuscript TeX checkpoint is never edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

import plot_drift_expression_context as source
import plot_drift_expression_context_v1 as v1


REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO / "manuscript/figures"
DATA_DIR = REPO / "manuscript/figures/data"


def plot_selection(ax: plt.Axes, aggregated: pd.DataFrame, dataset: str) -> None:
    d = aggregated[aggregated.dataset.eq(dataset)].copy()
    reference = d[~d.reference_panel.str.contains("boundary control", na=False)]
    max_rank = 2500

    # The lower-right half is the directly interpretable result region: the same
    # externally defined gene receives a better (smaller) rank with GRN guidance.
    ax.fill_between([1, max_rank], [1, max_rank], [1, 1], color="#E9F2EF", zorder=0)
    ax.plot([1, max_rank], [1, max_rank], color="#7D868A", linewidth=0.7,
            linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(reference.nogrn_rank, reference.grn_rank, s=11, facecolor="#AEB6BA",
               edgecolor="none", alpha=0.58, zorder=2)

    if dataset == "pancreas":
        v1.source_label(ax, "Pancreas")
        targets = [
            ("NEUROG3", "neurog3_late", 3, v1.GRN,
             "NEUROG3\nendocrine regulator", (8, 20)),
            ("SLC30A8", "sc_beta", 6, v1.NO_GRN,
             "SLC30A8\nterminal-marker control", (8, 20)),
        ]
    else:
        v1.source_label(ax, "Zebrafish")
        targets = [
            ("MYOD1", "adaxial_cells", 9, v1.GRN,
             "MYOD1\nmyogenic regulator", (-8, -20)),
            ("PCDH8", "somites", 9, v1.PURPLE,
             "PCDH8\nsomite-module gene", (-8, 20)),
        ]

    for gene, lineage, time, color, label, offset in targets:
        q = d[(d.gene == gene) & (d.lineage == lineage) & (d.time == time)]
        if q.empty:
            continue
        row = q.iloc[0]
        ax.scatter(row.nogrn_rank, row.grn_rank, s=30, facecolor=color,
                   edgecolor="white", linewidth=0.55, zorder=4)
        ax.annotate(label, (row.nogrn_rank, row.grn_rank), xytext=offset,
                    textcoords="offset points", fontsize=5.9, color=v1.OBSERVED,
                    va="center", ha="left" if offset[0] >= 0 else "right",
                    linespacing=1.05,
                    arrowprops=dict(arrowstyle="-", color="#727A7E", lw=0.45))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max_rank)
    ax.set_ylim(1, max_rank)
    ax.set_xlabel("No-GRN decoded-drift rank (1 is best)")
    ax.set_ylabel("Genuine-GRN decoded-drift rank (1 is best)")
    ax.set_xticks([1, 20, 100, 500, 2000])
    ax.set_xticklabels(["1", "20", "100", "500", "2000"])
    ax.set_yticks([1, 20, 100, 500, 2000])
    ax.set_yticklabels(["1", "20", "100", "500", "2000"])
    v1.clean_axis(ax, "both")


def plot_figure2(aggregated: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame,
                 rank_long: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(v1.WIDTH_IN, 6.75))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[0.92, 1.25], hspace=0.46, wspace=0.34,
                    left=0.105, right=0.985, top=0.925, bottom=0.09)
    ax_a = fig.add_subplot(grid[0, 0]); plot_selection(ax_a, aggregated, "pancreas")
    ax_b = fig.add_subplot(grid[0, 1]); plot_selection(ax_b, aggregated, "zebrafish")
    ax_c, _ = v1.plot_exemplar_panel(fig, grid[1, 0], raw, summary, rank_long, "pancreas")
    ax_d, _ = v1.plot_exemplar_panel(fig, grid[1, 1], raw, summary, rank_long, "zebrafish")
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        v1.panel_label(ax, label)
    handles = v1.condition_handles() + [
        v1.Line2D([0], [0], marker="o", color=v1.OBSERVED, markerfacecolor="white",
                  markeredgecolor=v1.OBSERVED, linewidth=0.9, label="Observed median and IQR"),
        v1.Patch(facecolor=v1.HELDOUT, edgecolor="none", label="Held-out time"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.2)
    v1.save(fig, FIGURE_DIR / "figure2_guidance_beyond_vae_v2")


def main() -> None:
    v1.register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    p_screen, p_rank, p_raw, p_summary = source.load_pancreas()
    z_screen, z_rank, z_raw, z_summary = source.load_zebrafish()
    aggregated = source.aggregate_screen(pd.concat([p_screen, z_screen], ignore_index=True))
    rank_long = pd.concat([p_rank.assign(readout="drift"), z_rank], ignore_index=True)
    raw = pd.concat([p_raw, z_raw], ignore_index=True)
    summary = pd.concat([p_summary, z_summary], ignore_index=True)
    plot_figure2(aggregated, raw, summary, rank_long)

    selection = aggregated[["dataset", "gene", "lineage", "lineage_display", "time",
                            "reference_panel", "nogrn_rank", "grn_rank", "n_seeds"]].copy()
    selection.to_csv(DATA_DIR / "figure2_v2_selection.csv", index=False)
    manifest = {
        "version": "v2 selection-panel revision",
        "argument": "GRN guidance adds lineage-specific regulatory attribution to a VAE baseline",
        "selection_rule": "external reference programs define the universe; biological role defines focal exemplars",
        "pancreas_exemplar": "NEUROG3, predefined endocrine-induction regulator",
        "zebrafish_exemplar": "MYOD1, predefined myogenic regulator",
        "figure_width_mm": 180,
        "png_ppi": v1.PNG_DPI,
        "font": "Open Sauce One",
        "outputs": ["figure2_guidance_beyond_vae_v2.pdf",
                    "figure2_guidance_beyond_vae_v2.svg",
                    "figure2_guidance_beyond_vae_v2.png",
                    "figure2_v2_selection.csv"],
    }
    (DATA_DIR / "drift_expression_context_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
