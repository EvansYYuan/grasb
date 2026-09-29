#!/usr/bin/env python
"""Render the v2 candidate-screen panels with two-sided failure-mode evidence."""

from __future__ import annotations

from copy import deepcopy
import json

from matplotlib.lines import Line2D
import pandas as pd

import plot_figure2_ab_v1 as base


REPO = base.REPO
FIGURE_DIR = base.FIGURE_DIR
DATA_DIR = base.DATA_DIR
INPUT = base.INPUT
OUTPUT_STEM = FIGURE_DIR / "figure2_ab_grn_added_v2"

CANDIDATES = deepcopy(base.CANDIDATES)

for spec in CANDIDATES["pancreas"]:
    if spec["gene"] in {"GCG", "INS"} and spec.get("favored") == "no GRN":
        spec["observed_expression_percentile"] = 1.0
        spec["evidence_class"] = "highly expressed terminal hormone"
    if spec["gene"] == "INS" and spec.get("favored") == "no GRN":
        spec["text"] = (2.65, 155)
        spec.pop("connector", None)

# Two additional pancreatic examples broaden the upper-half evidence from
# terminal hormones to abundant mature-state markers. Expression percentiles
# come from the held-out populations in figure2_reference_screen_aggregated.csv.
CANDIDATES["pancreas"].insert(
    2,
    dict(
        gene="ADRA2A", lineage="sc_ec", time=3, favored="no GRN",
        label=r"$\it{ADRA2A}$" + "\nabundant EC-like receptor",
        text=(2.1, 37),
        observed_expression_percentile=0.9745,
        evidence_class="highly expressed state marker",
    ),
)
CANDIDATES["pancreas"].insert(
    3,
    dict(
        gene="NKX6-1", lineage="sc_beta", time=3, favored="no GRN",
        label=r"$\mathit{NKX6{-}1}$" + "\nmature beta-cell identity TF",
        text=(58, 500), ha="center", connector=(58, 335),
        observed_expression_percentile=0.9766,
        evidence_class="highly expressed mature-state marker",
    ),
)

# Four zebrafish counterparts span terminal output, proliferation-associated,
# protease, and epithelial state markers preferred by no GRN.
CANDIDATES["zebrafish"].insert(
    0,
    dict(
        gene="APOC1", lineage="pancreatic_intestinal_endoderm", time=9,
        favored="no GRN", label=r"$\it{APOC1}$" + "\nendoderm apolipoprotein",
        text=(100, 3000), connector=(86, 2450),
        observed_expression_percentile=0.9955,
        evidence_class="highly expressed terminal output gene",
    ),
)
CANDIDATES["zebrafish"].insert(
    1,
    dict(
        gene="STMN1A", lineage="pancreatic_intestinal_endoderm", time=9,
        favored="no GRN", label=r"$\it{STMN1A}$" + "\nproliferation-associated gene",
        text=(115, 900), connector=(103, 1250),
        observed_expression_percentile=0.9865,
        evidence_class="highly expressed state-associated gene",
    ),
)
CANDIDATES["zebrafish"].insert(
    2,
    dict(
        gene="CTSLB", lineage="prechordal_plate", time=7,
        favored="no GRN", label=r"$\it{CTSLB}$" + "\nprechordal protease",
        text=(13, 590), ha="center",
        observed_expression_percentile=0.8565,
        evidence_class="terminal-state effector",
    ),
)
CANDIDATES["zebrafish"].insert(
    3,
    dict(
        gene="KRT18", lineage="prechordal_plate", time=7,
        favored="no GRN", label=r"$\it{KRT18}$" + "\nabundant epithelial marker",
        text=(25.5, 235), ha="right",
        observed_expression_percentile=0.9695,
        evidence_class="highly expressed state marker",
    ),
)

# base.plot_panel resolves candidates from its own module namespace.
base.CANDIDATES = CANDIDATES


def plot_panel(ax, selection, dataset, letter):
    """Render with full boundary markers; base v1 clips points at x=1."""
    selected = base.plot_panel(ax, selection, dataset, letter)
    for row in selected:
        favored = row.get("favored", "genuine GRN")
        color = base.NO_GRN_HIGHLIGHT if favored == "no GRN" else base.GRN_HIGHLIGHT
        ax.scatter(
            row["nogrn_rank"], row["grn_rank"], s=31, facecolor=color,
            edgecolor="white", linewidth=0.55, zorder=4, clip_on=False,
        )
    return selected


def add_rank_key(fig, anchor: tuple[float, float]) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=base.NO_GRN_HIGHLIGHT, markeredgecolor="white",
               markeredgewidth=0.45, label="favored by no-GRN"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.2,
               markerfacecolor=base.GRN_HIGHLIGHT, markeredgecolor="white",
               markeredgewidth=0.45, label="favored by genuine-GRN"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=anchor,
        ncol=2, frameon=False, fontsize=6.0, columnspacing=1.4,
        handletextpad=0.45,
    )


def main() -> None:
    base.style.register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(INPUT)

    fig, axes = base.plt.subplots(1, 2, figsize=(base.style.WIDTH_IN, 3.05))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.25, top=0.90, wspace=0.34)
    selected = []
    selected.extend(plot_panel(axes[0], selection, "pancreas", "a"))
    selected.extend(plot_panel(axes[1], selection, "zebrafish", "b"))
    add_rank_key(fig, (0.5, 0.012))
    base.style.save(fig, OUTPUT_STEM)

    pd.DataFrame(selected).to_csv(DATA_DIR / "figure2_ab_v2_candidates.csv", index=False)
    manifest = {
        "version": "Figure 2a-b candidate-screen review v2",
        "source": str(INPUT.relative_to(REPO)),
        "expression_evidence_source": (
            "manuscript/figures/data/figure2_reference_screen_aggregated.csv"
        ),
        "argument": (
            "genuine-GRN promotion of regulatory and lineage-associated genes is contrasted "
            "with no-GRN preference for terminal outputs and highly expressed state genes"
        ),
        "v2_additions": {
            "pancreas": ["ADRA2A", "NKX6-1"],
            "zebrafish": ["APOC1", "STMN1A", "CTSLB", "KRT18"],
        },
        "candidate_colors": {
            "favored by genuine GRN": base.GRN_HIGHLIGHT,
            "favored by no GRN": base.NO_GRN_HIGHLIGHT,
        },
        "outputs": [f"{OUTPUT_STEM.name}.{ext}" for ext in ("pdf", "svg", "png")]
        + ["figure2_ab_v2_candidates.csv"],
    }
    (DATA_DIR / "figure2_ab_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
