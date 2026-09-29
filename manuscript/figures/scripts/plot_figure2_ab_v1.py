#!/usr/bin/env python
"""Render only the candidate-screen panels for the revised Figure 2.

The panels compare no-GRN and genuine-GRN decoded-drift ranks within fixed
external reference programs. They intentionally contain no condition legend:
the two conditions are already named by the axes. Color identifies which model
preferentially ranks each annotated example.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import plot_drift_expression_context_v1 as style


REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO / "manuscript/figures"
DATA_DIR = REPO / "manuscript/figures/data"
INPUT = DATA_DIR / "figure2_v2_selection.csv"
OUTPUT_STEM = FIGURE_DIR / "figure2_ab_grn_added_v1"
GRN_HIGHLIGHT = style.GRN
NO_GRN_HIGHLIGHT = style.NO_GRN


# Candidate identity is fixed explicitly rather than selected at render time.
# Most represented cell types use only one or two clean exemplars. Additional
# central-bottom genes are included when they provide a direct regulatory
# interpretation or a biologically clean improvement in genuine-GRN rank.
# Label positions are in data coordinates on log axes.
CANDIDATES = {
    "pancreas": [
        # Above-diagonal output genes connect this overview directly to the
        # no-GRN failure mode resolved in Figure 3.
        dict(gene="GCG", lineage="sc_alpha", time=3, favored="no GRN",
             label=r"$\it{GCG}$" + "\nalpha-cell hormone", text=(2.0, 1050),
             connector=(1.55, 1420)),
        dict(gene="INS", lineage="sc_beta", time=6, favored="no GRN",
             label=r"$\it{INS}$" + "\nbeta-cell hormone", text=(3.4, 300),
             connector=(2.7, 225)),
        dict(gene="NEUROG3", lineage="neurog3_late", time=3,
             label=r"$\it{NEUROG3}$" + "\nlate endocrine progenitor TF", text=(250, 1.7), ha="right"),
        dict(gene="FEV", lineage="fev_high_isl_low", time=3,
             label=r"$\it{FEV}$" + "\nFEV-high endocrine progenitor TF", text=(1650, 6.1), ha="right"),
        dict(gene="SST", lineage="sst_hhex", time=6,
             label=r"$\it{SST}$" + "\ndelta-like hormone", text=(1450, 18), ha="right",
             connector=(1530, 13.5)),
        dict(gene="PAX4", lineage="neurog3_mid", time=3,
             label=r"$\it{PAX4}$" + "\nmid endocrine progenitor TF", text=(1650, 155), ha="right"),
        dict(gene="ARX", lineage="sc_alpha", time=6,
             label=r"$\it{ARX}$" + "\nalpha-cell fate TF", text=(250, 42)),
        dict(gene="DDC", lineage="sc_ec", time=3,
             label=r"$\it{DDC}$" + "\nEC-like serotonin enzyme", text=(165, 10), ha="right"),
    ],
    "zebrafish": [
        dict(gene="RIPPLY1", lineage="somites", time=9,
             label=r"$\it{RIPPLY1}$" + "\nsomite-boundary regulator", text=(62, 31), ha="right",
             connector=(76, 24.5)),
        dict(gene="CDKN1CA", lineage="adaxial_cells", time=9,
             label=r"$\it{CDKN1CA}$" + "\nadaxial-cell cycle regulator", text=(38, 6.2)),
        dict(gene="MYF5", lineage="somites", time=9,
             label=r"$\it{MYF5}$" + "\nsomite myogenic TF", text=(2.1, 4.5),
             connector=(15.5, 5.3)),
        dict(gene="TA", lineage="notochord", time=9,
             label=r"$\it{TA}$" + "\nnotochord-fate TF", text=(1500, 12.5), ha="right"),
        dict(gene="PMP22B", lineage="notochord", time=9,
             label=r"$\it{PMP22B}$" + "\nnotochord membrane candidate", text=(1500, 2.7), ha="right",
             connector=(1700, 3.8)),
        dict(gene="MYOD1", lineage="adaxial_cells", time=9,
             label=r"$\it{MYOD1}$" + "\nadaxial myogenic TF", text=(1400, 25), ha="right"),
        dict(gene="PCDH8", lineage="somites", time=9,
             label=r"$\it{PCDH8}$" + "\nsomite adhesion gene", text=(950, 58), ha="right",
             connector=(1080, 49)),
        dict(gene="TBX16", lineage="adaxial_cells", time=9,
             label=r"$\it{TBX16}$" + "\nparaxial-mesoderm TF", text=(1400, 115), ha="right",
             connector=(1510, 78)),
    ],
}


def candidate_row(data: pd.DataFrame, spec: dict) -> pd.Series:
    q = data[
        data.gene.eq(spec["gene"])
        & data.lineage.eq(spec["lineage"])
        & data.time.eq(float(spec["time"]))
    ]
    if len(q) != 1:
        raise ValueError(f"Expected one row for {spec}, found {len(q)}")
    return q.iloc[0]


def plot_panel(ax: plt.Axes, selection: pd.DataFrame, dataset: str, letter: str) -> list[dict]:
    data = selection[selection.dataset.eq(dataset)].copy()
    reference = data[~data.reference_panel.str.contains("boundary control", na=False)]
    max_rank = 2500

    ax.fill_between([1, max_rank], [1, max_rank], [1, 1], color="#E9F2EF", zorder=0)
    ax.plot([1, max_rank], [1, max_rank], color="#7D868A", linewidth=0.7,
            linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(reference.nogrn_rank, reference.grn_rank, s=11, facecolor="#AEB6BA",
               edgecolor="none", alpha=0.58, zorder=2)

    selected_rows = []
    for spec in CANDIDATES[dataset]:
        row = candidate_row(data, spec)
        favored = spec.get("favored", "genuine GRN")
        selected_rows.append({"favored": favored, **spec, **row.to_dict()})
        color = NO_GRN_HIGHLIGHT if favored == "no GRN" else GRN_HIGHLIGHT
        ax.scatter(row.nogrn_rank, row.grn_rank, s=31, facecolor=color,
                   edgecolor="white", linewidth=0.55, zorder=4)
        tx, ty = spec["text"]
        if "connector" in spec:
            cx, cy = spec["connector"]
            ax.plot([row.nogrn_rank, cx], [row.grn_rank, cy], color="#667176",
                    linewidth=0.65, solid_capstyle="round", zorder=3)
        ax.annotate(spec["label"], (tx, ty), fontsize=5.6, color=style.OBSERVED,
                    va="center", ha=spec.get("ha", "left"), linespacing=1.05,
                    zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, max_rank)
    ax.set_ylim(1, 3150)
    ax.set_xlabel("No-GRN decoded-drift rank (1 is best)")
    ax.set_ylabel("Genuine-GRN decoded-drift rank (1 is best)")
    ticks = [1, 20, 100, 500, 2000]
    labels = ["1", "20", "100", "500", "2000"]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    style.clean_axis(ax, "both")

    # Keep identifiers outside the data-dense region and above the y=2000 gridline.
    ax.text(-0.14, 1.025, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom", clip_on=False)
    ax.text(1.12, 2200, "Pancreas" if dataset == "pancreas" else "Zebrafish",
            transform=ax.transData, fontsize=7.2, fontweight="bold",
            color="#495156", ha="left", va="bottom")
    return selected_rows


def main() -> None:
    style.register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(INPUT)

    fig, axes = plt.subplots(1, 2, figsize=(style.WIDTH_IN, 3.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.20, top=0.90, wspace=0.34)
    selected = []
    selected.extend(plot_panel(axes[0], selection, "pancreas", "a"))
    selected.extend(plot_panel(axes[1], selection, "zebrafish", "b"))
    style.save(fig, OUTPUT_STEM)

    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(DATA_DIR / "figure2_ab_v1_candidates.csv", index=False)
    manifest = {
        "version": "Figure 2a-b candidate-screen review v1",
        "source": str(INPUT.relative_to(REPO)),
        "argument": (
            "genes promoted by genuine-GRN guidance contrasted with prominent output genes "
            "preferentially ranked by no GRN"
        ),
        "selection_rule": (
            "primarily one or two far-right clean genuine-GRN exemplars per represented cell "
            "type, plus RIPPLY1, CDKN1CA, and MYF5 as biologically supported central-bottom "
            "cases; GCG and INS are Figure 3-matched above-diagonal no-GRN exemplars"
        ),
        "candidate_colors": {
            "favored by genuine GRN": GRN_HIGHLIGHT,
            "favored by no GRN": NO_GRN_HIGHLIGHT,
        },
        "legend": "omitted; conditions are named by the axes",
        "outputs": [f"{OUTPUT_STEM.name}.{ext}" for ext in ("pdf", "svg", "png")]
                   + ["figure2_ab_v1_candidates.csv"],
    }
    (DATA_DIR / "figure2_ab_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
