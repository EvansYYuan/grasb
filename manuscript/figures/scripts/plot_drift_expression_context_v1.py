#!/usr/bin/env python
"""Journal-style v1 of the observed-expression/drift manuscript figures.

Figure 2 asks where GRN guidance adds regulatory attribution beyond the frozen-VAE baseline.
Figure 3 asks whether local decoded-drift gains propagate to integrated expression transport.
Pancreatic and zebrafish measurements are never pooled within a subplot.

Run with:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/plot_drift_expression_context_v1.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import plot_drift_expression_context as source


REPO = Path(__file__).resolve().parents[3]
FIGURE_DIR = REPO / "manuscript/figures"
DATA_DIR = REPO / "manuscript/figures/data"
FONT_DIR = REPO / "manuscript/figures/assets/fonts"

WIDTH_IN = 180 / 25.4
PNG_DPI = 600

NO_GRN = "#2F668F"
GRN = "#E1645D"
OBSERVED = "#20262A"
TEAL = "#2A948B"
PURPLE = "#7A5AA6"
GRAY = "#98A1A6"
LIGHT_GRAY = "#D8DDE0"
HELDOUT = "#E8E8E8"


def register_fonts() -> None:
    for path in sorted(FONT_DIR.glob("OpenSauceOne-*.ttf")):
        font_manager.fontManager.addfont(path)
    mpl.rcParams.update({
        "font.family": "Open Sauce One",
        "font.size": 7.0,
        "axes.labelsize": 7.3,
        "axes.titlesize": 7.3,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.055, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", clip_on=False)


def source_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.015, 0.975, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=6.4, fontweight="bold", color="#3D4549")


def gene_label(ax: plt.Axes, text: str, x: float = 0.02, y: float = 0.92) -> None:
    ax.text(x, y, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=7.0, fontstyle="italic", color=OBSERVED)


def clean_axis(ax: plt.Axes, grid: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E8EBEC", linewidth=0.5, zorder=0)


def condition_handles() -> list:
    return [
        Line2D([0], [0], marker="o", color=NO_GRN, markerfacecolor="white",
               markeredgecolor=NO_GRN, linewidth=1.0, linestyle=(0, (3, 2)),
               label="No GRN (VAE baseline)"),
        Line2D([0], [0], marker="o", color=GRN, markerfacecolor=GRN,
               markeredgecolor=GRN, linewidth=1.0, label="Genuine GRN"),
    ]


def plot_systematic(ax: plt.Axes, aggregated: pd.DataFrame, dataset: str) -> None:
    d = aggregated[aggregated.dataset.eq(dataset)].copy()
    primary = d[~d.reference_panel.str.contains("boundary control", na=False)]
    ax.scatter(primary.expression_percentile, primary.rank_score_gain,
               s=10, facecolor="#AEB6BA", edgecolor="none", alpha=0.55, zorder=2)
    bins = pd.cut(primary.expression_percentile, np.linspace(0, 1, 6), include_lowest=True)
    median = (primary.groupby(bins, observed=True)
              .agg(x=("expression_percentile", "median"),
                   y=("rank_score_gain", "median")).dropna())
    ax.plot(median.x, median.y, color=OBSERVED, marker="o", markersize=2.8,
            linewidth=1.05, zorder=3, label="bin median")
    ax.axhline(0, color="#6E777C", linewidth=0.65, zorder=1)

    if dataset == "pancreas":
        targets = [("NEUROG3", "neurog3_late", 3, GRN, (3, 5)),
                   ("SLC30A8", "sc_beta", 6, NO_GRN, (3, -9))]
        source_label(ax, "Pancreas")
    else:
        targets = [("MYOD1", "adaxial_cells", 9, GRN, (-30, 6)),
                   ("PCDH8", "somites", 9, PURPLE, (-28, -10))]
        source_label(ax, "Zebrafish")
    for gene, lineage, time, color, offset in targets:
        q = d[(d.gene == gene) & (d.lineage == lineage) & (d.time == time)]
        if q.empty:
            continue
        row = q.iloc[0]
        ax.scatter(row.expression_percentile, row.rank_score_gain, s=25,
                   facecolor=color, edgecolor="white", linewidth=0.55, zorder=4)
        ax.annotate(gene, (row.expression_percentile, row.rank_score_gain),
                    xytext=offset, textcoords="offset points", fontsize=6.3,
                    fontstyle="italic", color=OBSERVED,
                    arrowprops=dict(arrowstyle="-", color="#727A7E", lw=0.45))
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Observed-expression percentile")
    ax.set_ylabel("GRN gain in decoded-drift rank score")
    clean_axis(ax, "y")
    ax.legend(frameon=False, loc="lower left", handlelength=1.5)


def plot_expression_distribution(ax: plt.Axes, raw: pd.DataFrame, summary: pd.DataFrame,
                                 times: list[float], heldout: set[float], color: str,
                                 x_label: str) -> None:
    for t in times:
        if t in heldout:
            width = 0.28 if len(times) <= 8 else 0.16
            ax.axvspan(t - width, t + width, color=HELDOUT, zorder=0)
    positions, arrays = [], []
    for t in times:
        values = raw.loc[raw.time.eq(t), "expression"].dropna().to_numpy()
        if len(values) >= 2:
            positions.append(t); arrays.append(values)
        elif len(values) == 1:
            ax.scatter(t, values[0], s=8, color=OBSERVED, zorder=4)
    if arrays:
        parts = ax.violinplot(arrays, positions=positions, widths=0.46,
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.18)
    s = summary[summary.n_cells.gt(0)].sort_values("time")
    ax.plot(s.time, s.observed_median, color=OBSERVED, linewidth=0.95, zorder=4)
    ax.vlines(s.time, s.observed_q25, s.observed_q75, color=OBSERVED, linewidth=0.8, zorder=4)
    ax.scatter(s.time, s.observed_median, s=13, facecolor="white", edgecolor=OBSERVED,
               linewidth=0.65, zorder=5)
    ax.set_xticks(times)
    if any(float(t) != int(t) for t in times):
        ax.set_xticklabels([f"{t:g}" for t in times], rotation=45, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Observed normalized expression")
    clean_axis(ax, "y")


def plot_rank_compass(ax: plt.Axes, rank_rows: pd.DataFrame, topk: int,
                      seed_resolved: bool) -> None:
    ax.axvline(topk, color="#7F888D", linewidth=0.6, linestyle=(0, (2, 2)))
    if seed_resolved:
        for seed, q in rank_rows.groupby("seed"):
            by = q.set_index("condition")
            x_no, x_grn = by.loc["no GRN", "rank"], by.loc["genuine GRN", "rank"]
            ax.plot([x_no, x_grn], [seed, seed], color="#AEB5B8", linewidth=0.75, zorder=1)
            ax.scatter(x_no, seed, s=18, facecolor="white", edgecolor=NO_GRN,
                       linewidth=0.8, zorder=3)
            ax.scatter(x_grn, seed, s=18, facecolor=GRN, edgecolor=GRN,
                       linewidth=0.8, zorder=3)
        ax.set_yticks([0, 1, 2], ["seed 0", "seed 1", "seed 2"])
    else:
        by = rank_rows.set_index("condition")
        x_no, x_grn = by.loc["no GRN", "rank"], by.loc["genuine GRN", "rank"]
        ax.plot([x_no, x_grn], [0, 0], color="#AEB5B8", linewidth=0.85)
        ax.scatter(x_no, 0, s=20, facecolor="white", edgecolor=NO_GRN,
                   linewidth=0.9, zorder=3)
        ax.scatter(x_grn, 0, s=20, facecolor=GRN, edgecolor=GRN,
                   linewidth=0.9, zorder=3)
        ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
        ax.text(x_no, 0.16, str(int(x_no)), color=NO_GRN, ha="center", fontsize=5.8)
        ax.text(x_grn, -0.18, str(int(x_grn)), color=GRN, ha="center", va="top", fontsize=5.8)
    ax.set_xscale("log")
    ax.set_xlabel("Decoded-drift rank (rank 1 is best)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#ECEEEF", linewidth=0.45)


def plot_exemplar_panel(fig: plt.Figure, spec, raw: pd.DataFrame, summary: pd.DataFrame,
                        rank_long: pd.DataFrame, dataset: str) -> tuple[plt.Axes, plt.Axes]:
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=spec, height_ratios=[2.65, 1.0], hspace=0.67)
    top = fig.add_subplot(sub[0]); bottom = fig.add_subplot(sub[1])
    if dataset == "pancreas":
        r = raw[(raw.dataset == dataset) & (raw.gene == "NEUROG3")]
        s = summary[(summary.dataset == dataset) & (summary.gene == "NEUROG3")]
        plot_expression_distribution(top, r, s, list(range(8)), {3, 6}, TEAL, "Differentiation day")
        q = rank_long[(rank_long.dataset == dataset) & (rank_long.gene == "NEUROG3") &
                      (rank_long.lineage == "neurog3_late") & (rank_long.stage == 3)]
        plot_rank_compass(bottom, q, 15, False)
        source_label(top, "Pancreas")
        gene_label(top, "NEUROG3")
    else:
        r = raw[(raw.dataset == dataset) & (raw.gene == "MYOD1")]
        s = summary[(summary.dataset == dataset) & (summary.gene == "MYOD1")]
        times = [8.0, 9.0, 10.0, 11.0, 12.0]
        plot_expression_distribution(top, r, s, times, {9.0}, TEAL, "Hours post-fertilization")
        q = rank_long[(rank_long.dataset == dataset) & (rank_long.gene == "MYOD1") &
                      (rank_long.lineage == "adaxial_cells") & (rank_long.stage == 9) &
                      (rank_long.readout == "drift")]
        plot_rank_compass(bottom, q, 50, True)
        source_label(top, "Zebrafish")
        gene_label(top, "MYOD1")
    return top, bottom


def plot_figure2(aggregated: pd.DataFrame, raw: pd.DataFrame, summary: pd.DataFrame,
                 rank_long: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH_IN, 6.75))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[0.88, 1.25], hspace=0.46, wspace=0.34,
                    left=0.105, right=0.985, top=0.925, bottom=0.09)
    ax_a = fig.add_subplot(grid[0, 0]); plot_systematic(ax_a, aggregated, "pancreas")
    ax_b = fig.add_subplot(grid[0, 1]); plot_systematic(ax_b, aggregated, "zebrafish")
    ax_c, _ = plot_exemplar_panel(fig, grid[1, 0], raw, summary, rank_long, "pancreas")
    ax_d, _ = plot_exemplar_panel(fig, grid[1, 1], raw, summary, rank_long, "zebrafish")
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        panel_label(ax, label)
    handles = condition_handles() + [
        Line2D([0], [0], marker="o", color=OBSERVED, markerfacecolor="white",
               markeredgecolor=OBSERVED, linewidth=0.9, label="Observed median and IQR"),
        Patch(facecolor=HELDOUT, edgecolor="none", label="Held-out time"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.985), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.2)
    save(fig, FIGURE_DIR / "figure2_guidance_beyond_vae_v1")


def plot_pancreas_trajectories(fig: plt.Figure, spec, trajectories: pd.DataFrame) -> plt.Axes:
    sub = GridSpecFromSubplotSpec(3, 1, subplot_spec=spec, hspace=0.10)
    first = None
    for i, gene in enumerate(["NEUROG3", "NEUROD1", "SLC30A8"]):
        ax = fig.add_subplot(sub[i]); first = ax if first is None else first
        for day in [3, 6]:
            ax.axvspan(day - 0.15, day + 0.15, color=HELDOUT, zorder=0)
        for condition, color, ls, marker, face in [
            ("observed", OBSERVED, "-", "o", OBSERVED),
            ("no GRN", NO_GRN, (0, (3, 2)), "o", "white"),
            ("genuine GRN", GRN, "-", "o", GRN),
        ]:
            q = trajectories[(trajectories.gene == gene) & (trajectories.condition == condition)]
            ax.plot(q.time, q.expression, color=color, linestyle=ls, linewidth=0.9,
                    marker=marker, markersize=2.8, markerfacecolor=face,
                    markeredgecolor=color, markeredgewidth=0.55, label=condition)
        gene_label(ax, gene, 0.015, 0.90)
        clean_axis(ax, "y")
        ax.set_xlim(-0.25, 7.25)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Differentiation day")
        if i == 1:
            ax.set_ylabel("Mean normalized expression")
    first.text(0.985, 0.92, "Pancreas", transform=first.transAxes, ha="right", va="top",
               fontsize=6.4, fontweight="bold", color="#3D4549")
    return first


def plot_effect_map(ax: plt.Axes, effects: pd.DataFrame, dataset: str) -> None:
    d = effects[effects.dataset.eq(dataset)]
    ax.scatter(d.local_rank_score_gain, d.integrated_gain, s=9,
               facecolor="#AAB2B6", edgecolor="none", alpha=0.55)
    ax.axhline(0, color="#717A7F", linewidth=0.6)
    ax.axvline(0, color="#717A7F", linewidth=0.6)
    if dataset == "pancreas":
        labels = d[d.gene.isin(["NEUROG3", "SLC30A8"])]
        source_label(ax, "Pancreas")
        ax.set_ylabel("Δ trajectory MAE\n(no GRN − genuine GRN)")
    else:
        labels = d[((d.gene == "MYOD1") & (d.lineage == "adaxial_cells") & (d.time == 9)) |
                   ((d.gene == "PCDH8") & (d.lineage == "somites") & (d.time == 9))]
        source_label(ax, "Zebrafish")
        ax.set_ylabel("GRN gain in generated-expression rank score")
    for row in labels.itertuples():
        color = GRN if row.gene in {"NEUROG3", "MYOD1"} else PURPLE
        ax.scatter(row.local_rank_score_gain, row.integrated_gain, s=24,
                   color=color, edgecolor="white", linewidth=0.5, zorder=4)
        if row.gene == "MYOD1":
            offset = (5, 16)
        elif row.gene == "PCDH8":
            offset = (-42, 8)
        elif row.gene == "NEUROG3":
            offset = (4, 5)
        else:
            offset = (-30, 5)
        ax.annotate(row.gene, (row.local_rank_score_gain, row.integrated_gain),
                    xytext=offset, textcoords="offset points", fontsize=6.2,
                    fontstyle="italic")
    ax.set_xlabel("GRN gain in decoded-drift rank score")
    clean_axis(ax, "y")


def plot_flow_pair(fig: plt.Figure, spec, flow: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(spec)
    colors = {"MYOD1": TEAL, "PCDH8": PURPLE}
    positions = {("drift", "no GRN"): 0.0, ("drift", "genuine GRN"): 1.0,
                 ("expression", "no GRN"): 2.25, ("expression", "genuine GRN"): 3.25}
    for readout in ["drift", "expression"]:
        for gene in ["MYOD1", "PCDH8"]:
            q = flow[(flow.readout == readout) & (flow.gene == gene)]
            for seed, g in q.groupby("seed"):
                by = g.set_index("condition")
                y0, y1 = by.loc["no GRN", "rank"], by.loc["genuine GRN", "rank"]
                x0 = positions[(readout, "no GRN")]
                x1 = positions[(readout, "genuine GRN")]
                ax.plot([x0, x1], [y0, y1], color=colors[gene], alpha=0.62, linewidth=0.75)
                ax.scatter(x0, y0, s=14, facecolor="white", edgecolor=colors[gene], linewidth=0.65)
                ax.scatter(x1, y1, s=14, facecolor=colors[gene], edgecolor=colors[gene], linewidth=0.65)
    ax.axhline(50, xmin=0.0, xmax=0.31, color="#7D858A",
               linestyle=(0, (2, 2)), linewidth=0.55)
    ax.text(0.98, 50, "top 50", ha="right", va="bottom", fontsize=5.5, color="#656D71")
    ax.text(0.5, 0.97, "Decoded drift", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=6.4, fontweight="bold")
    ax.text(2.75, 0.97, "Generated expression", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=6.4, fontweight="bold")
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xlim(-0.18, 3.43)
    ax.set_xticks([0, 1, 2.25, 3.25], ["No GRN", "GRN", "No GRN", "GRN"])
    ax.set_ylabel("Rank (rank 1 is best)")
    clean_axis(ax, "y")
    ax.text(0.985, 0.04, "Zebrafish", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.4, fontweight="bold", color="#3D4549")
    handles = [Line2D([0], [0], color=colors[g], marker="o", label=g, linewidth=0.9,
                      markersize=3.5) for g in ["MYOD1", "PCDH8"]]
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=2,
              columnspacing=0.9, handlelength=1.2)
    return ax


def plot_figure3(trajectories: pd.DataFrame, effects: pd.DataFrame,
                 flow: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(WIDTH_IN, 6.35))
    grid = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.36,
                    left=0.11, right=0.985, top=0.925, bottom=0.095)
    ax_a = plot_pancreas_trajectories(fig, grid[0, 0], trajectories)
    ax_b = fig.add_subplot(grid[0, 1]); plot_effect_map(ax_b, effects, "pancreas")
    ax_c = fig.add_subplot(grid[1, 0]); plot_effect_map(ax_c, effects, "zebrafish")
    ax_d = plot_flow_pair(fig, grid[1, 1], flow)
    for ax, label in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
        panel_label(ax, label)
    handles = [
        Line2D([0], [0], marker="o", color=OBSERVED, markerfacecolor=OBSERVED,
               linewidth=0.9, label="Observed"),
        *condition_handles(),
        Patch(facecolor=HELDOUT, edgecolor="none", label="Held-out time"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.987), ncol=4,
               frameon=False, handlelength=1.8, columnspacing=1.15)
    save(fig, FIGURE_DIR / "figure3_local_vs_integrated_v1")


def save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    plt.close(fig)
    print(f"[write] {stem}.{{pdf,svg,png}}", flush=True)


def main() -> None:
    register_fonts()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    p_screen, p_rank, p_raw, p_summary = source.load_pancreas()
    z_screen, z_rank, z_raw, z_summary = source.load_zebrafish()
    screen = pd.concat([p_screen, z_screen], ignore_index=True)
    aggregated = source.aggregate_screen(screen)
    rank_long = pd.concat([p_rank.assign(readout="drift"), z_rank], ignore_index=True)
    raw = pd.concat([p_raw, z_raw], ignore_index=True)
    summary = pd.concat([p_summary, z_summary], ignore_index=True)
    trajectories = source.load_pancreas_trajectories()
    effects = pd.concat([
        source.pancreas_local_integrated(p_rank, trajectories),
        source.zebrafish_local_integrated(z_screen, z_rank),
    ], ignore_index=True)
    flow = source.exemplar_flow(z_rank)

    plot_figure2(aggregated, raw, summary, rank_long)
    plot_figure3(trajectories, effects, flow)

    exports = {
        "figure2_v1_systematic.csv": aggregated,
        "figure2_v1_exemplar_expression_raw.csv": raw,
        "figure2_v1_exemplar_expression_summary.csv": summary,
        "figure2_v1_exemplar_ranks.csv": rank_long[
            ((rank_long.dataset == "pancreas") & (rank_long.gene == "NEUROG3") &
             (rank_long.lineage == "neurog3_late") & (rank_long.stage == 3)) |
            ((rank_long.dataset == "zebrafish") & (rank_long.gene == "MYOD1") &
             (rank_long.lineage == "adaxial_cells") & (rank_long.stage == 9))],
        "figure3_v1_trajectories.csv": trajectories,
        "figure3_v1_local_vs_integrated.csv": effects,
        "figure3_v1_exemplar_flow.csv": flow,
    }
    for name, frame in exports.items():
        frame.to_csv(DATA_DIR / name, index=False)

    manifest = {
        "version": "v1 journal-style review",
        "figure_width_mm": 180,
        "png_ppi": PNG_DPI,
        "vector_outputs": ["PDF", "SVG"],
        "font": "Open Sauce One",
        "font_license": str(FONT_DIR / "Open_Sauce_One_OFL.txt"),
        "dataset_separation": "pancreas and zebrafish are never pooled within a subplot",
        "figure2_argument": "GRN guidance adds regulatory attribution beyond the VAE baseline",
        "figure3_argument": "local decoded-drift gain and integrated expression transport are distinct",
        "runtime": {"python": sys.version.split()[0], "executable": sys.executable,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "matplotlib": mpl.__version__},
        "outputs": [
            "figure2_guidance_beyond_vae_v1.pdf", "figure2_guidance_beyond_vae_v1.svg",
            "figure2_guidance_beyond_vae_v1.png", "figure3_local_vs_integrated_v1.pdf",
            "figure3_local_vs_integrated_v1.svg", "figure3_local_vs_integrated_v1.png",
            *exports.keys(),
        ],
    }
    (DATA_DIR / "drift_expression_context_v1_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("[done] v1 journal-style figures", flush=True)


if __name__ == "__main__":
    main()
