#!/usr/bin/env python
"""Build manuscript Figures 2 and 3 from frozen reference panels and raw expression.

The figures deliberately treat the no-GRN condition as a VAE-informed expression-pattern
baseline.  Genuine-GRN guidance is evaluated as additional regulatory attribution, not as a
requirement for every biologically recognizable gene.  Local decoded drift and integrated
generated expression are kept as separate readouts throughout.

Outputs include PDF/SVG/PNG figures, exact plotted-data CSVs, and a provenance manifest.  This
script never edits the manuscript TeX checkpoint.

Run with the repository's isolated plotting environment:
  conda run -n grasb-figures python \
      manuscript/figures/scripts/plot_drift_expression_context.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data"
DEFAULT_FIGURE_OUT = REPO / "manuscript/figures"
DEFAULT_DATA_OUT = REPO / "manuscript/figures/data"

PANCREAS_EXPR = DATA / "pancreatic_preprocessed.csv"
PANCREAS_META = DATA / "pancreatic_metadata.tsv"
PANCREAS_DRIFT = REPO / "results/per_type_drift.csv"
PANCREAS_OBS_TRAJ = REPO / "results/pancreas_observed_trajectory.csv"
PANCREAS_GRN_TRAJ = REPO / "results/pancreas_grasb_trajectory.csv"
PANCREAS_NOGRN_TRAJ = REPO / "results/pancreas_nogrn_trajectory.csv"

ZEBRAFISH_EXPR = DATA / "zebrafish_preprocessed.csv"
ZEBRAFISH_META = DATA / "zebrafish_metadata.csv"
ZEBRAFISH_GROUPS = DATA / "zebrafish_groups.csv"
ZEBRAFISH_SEGMENTS = REPO / "config/zebrafish/segment_map.csv"
ZEBRAFISH_MODULES = REPO / "config/zebrafish/published_modules.csv"
ZEBRAFISH_GENERATED = REPO / "zebrafish/output/corrected_audit/generated"

PANCREAS_MARKERS = {
    # Veres et al. Extended Data Table 2 positive identity markers, in-HVG subset.
    # Kept identical to q_biology_gsea.py, the evaluation's source of truth.
    "sc_beta": ["INS", "NKX6-1", "ISL1", "PAX4", "G6PC2", "NPTX2"],
    "sc_alpha": ["GCG", "ARX", "IRX2", "CD36", "ISL1"],
    "sc_ec": ["TPH1", "SLC18A1", "FEV", "DDC", "ADRA2A"],
    "sst_hhex": ["SST", "HHEX", "ISL1", "RBP4"],
    "neurog3_early": ["NEUROG3", "BTG2", "GADD45A"],
    "neurog3_mid": ["NEUROG3", "PAX4", "HES6"],
    "neurog3_late": ["NEUROG3", "NEUROD1", "FEV"],
    "prog_nkx61": ["NKX6-1", "PTF1A", "SOX9", "SPP1"],
    "phox2a": ["PHOX2A", "TPH1", "FEV", "KLK1", "ENC1"],
    "fev_high_isl_low": ["FEV", "PAX4", "ONECUT3"],
}

PANCREAS_DISPLAY = {
    "sc_beta": "beta-like", "sc_alpha": "alpha-like", "sc_ec": "EC-like",
    "sst_hhex": "delta-like", "neurog3_early": "early endocrine progenitor",
    "neurog3_mid": "mid endocrine progenitor", "neurog3_late": "late endocrine progenitor",
    "prog_nkx61": "NKX6-1+ progenitor", "phox2a": "PHOX2A+",
    "fev_high_isl_low": "FEV-high endocrine progenitor",
}

COLORS = {
    "grn": "#D55E5D", "nogrn": "#315A7D", "observed": "#202428",
    "pancreas": "#8B5FA8", "zebrafish": "#249A8D", "muted": "#90999F",
    "paper": "#FCFBF7", "grid": "#DDE1E2", "heldout": "#F3DDD7",
}

HELDOUT_PANCREAS = {3, 6}
HELDOUT_ZEBRAFISH = {5.3, 7.0, 9.0}
ZEBRAFISH_PRIMARY_HPF = {7.0, 9.0}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figure-out-dir", type=Path, default=DEFAULT_FIGURE_OUT)
    p.add_argument("--data-out-dir", type=Path, default=DEFAULT_DATA_OUT)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank_frame(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Deterministic descending ordinal rank with gene-symbol tie breaking."""
    out = frame[["gene", value]].copy()
    out["gene"] = out["gene"].astype(str).str.upper()
    out = out.sort_values([value, "gene"], ascending=[False, True], kind="mergesort")
    out = out.drop_duplicates("gene", keep="first").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["rank_score"] = 1.0 - (out["rank"] - 1) / max(len(out) - 1, 1)
    out["n_ranked"] = len(out)
    return out


def distribution_summary(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n_cells": int(len(values)),
        "observed_mean": float(values.mean()) if len(values) else np.nan,
        "observed_median": float(values.median()) if len(values) else np.nan,
        "observed_q25": float(values.quantile(0.25)) if len(values) else np.nan,
        "observed_q75": float(values.quantile(0.75)) if len(values) else np.nan,
        "detection_fraction": float((values > 0).mean()) if len(values) else np.nan,
    }


def load_pancreas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[load] pancreatic expression", flush=True)
    expr = pd.read_csv(PANCREAS_EXPR, index_col=0)
    meta = pd.read_csv(PANCREAS_META, sep="\t", index_col="library.barcode").reindex(expr.index)
    if meta.Assigned_cluster.isna().any():
        raise ValueError("Pancreatic expression contains barcodes absent from metadata")
    if not np.array_equal(expr["time"].to_numpy(), meta.CellWeek.to_numpy()):
        raise ValueError("Pancreatic expression time does not match CellWeek")

    genes = [c for c in expr.columns if c != "time"]
    drift = pd.read_csv(PANCREAS_DRIFT)
    rank_rows: list[pd.DataFrame] = []
    screen_rows: list[dict] = []
    for cell_type, markers in PANCREAS_MARKERS.items():
        for stage in sorted(HELDOUT_PANCREAS):
            mask = meta.Assigned_cluster.eq(cell_type) & meta.CellWeek.eq(stage)
            x = expr.loc[mask, genes]
            if x.empty:
                continue
            means = x.mean(axis=0)
            detect = x.gt(0).mean(axis=0)
            mean_pct = means.rank(method="average", pct=True)
            detect_pct = detect.rank(method="average", pct=True)
            by_condition = {}
            for condition, tag in [("genuine GRN", "latent_grn"), ("no GRN", "latent_nogrn")]:
                unit = drift[(drift.tag == tag) & (drift.cell_type == cell_type) &
                             (drift.stage == stage)]
                ranked = rank_frame(unit, "drift")
                ranked.insert(0, "condition", condition)
                ranked.insert(0, "stage", stage)
                ranked.insert(0, "lineage", cell_type)
                ranked.insert(0, "dataset", "pancreas")
                ranked.insert(4, "seed", np.nan)
                ranked = ranked.rename(columns={"drift": "value"})
                rank_rows.append(ranked)
                by_condition[condition] = ranked.set_index("gene")
            screen_genes = list(markers)
            if cell_type == "sc_beta":
                # Prespecified boundary example from the figure plan.  It is not silently folded
                # into the final EDT2 scoring panel; provenance stays explicit in the export.
                screen_genes.append("SLC30A8")
            for gene in screen_genes:
                gene = gene.upper()
                if gene not in means.index or any(gene not in r.index for r in by_condition.values()):
                    continue
                grn = by_condition["genuine GRN"].loc[gene]
                nogrn = by_condition["no GRN"].loc[gene]
                screen_rows.append({
                    "dataset": "pancreas", "gene": gene, "lineage": cell_type,
                    "lineage_display": PANCREAS_DISPLAY[cell_type], "time": stage,
                    "heldout": True, "n_cells": int(mask.sum()),
                    "observed_mean": means[gene], "detection_fraction": detect[gene],
                    "expression_percentile": mean_pct[gene],
                    "detection_percentile": detect_pct[gene],
                    "grn_rank": int(grn["rank"]), "nogrn_rank": int(nogrn["rank"]),
                    "grn_rank_score": grn["rank_score"],
                    "nogrn_rank_score": nogrn["rank_score"],
                    "rank_gain": int(nogrn["rank"] - grn["rank"]),
                    "rank_score_gain": grn["rank_score"] - nogrn["rank_score"],
                    "seed": np.nan,
                    "reference_panel": ("prespecified Veres terminal-gene boundary control"
                                        if gene == "SLC30A8" else "Veres EDT2"),
                })

    raw_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    focal = expr[["time", "NEUROG3"]].copy()
    focal["lineage"] = meta.Assigned_cluster.to_numpy()
    focal = focal[focal.lineage.eq("neurog3_late")]
    for time in range(8):
        values = focal.loc[focal.time.eq(time), "NEUROG3"]
        summary_rows.append({"dataset": "pancreas", "gene": "NEUROG3",
                             "lineage": "neurog3_late", "time": time,
                             "heldout": time in HELDOUT_PANCREAS, **distribution_summary(values)})
        raw_rows.append(pd.DataFrame({
            "dataset": "pancreas", "gene": "NEUROG3", "lineage": "neurog3_late",
            "time": time, "heldout": time in HELDOUT_PANCREAS,
            "cell": values.index.astype(str), "expression": values.to_numpy(),
        }))
    return (pd.DataFrame(screen_rows), pd.concat(rank_rows, ignore_index=True),
            pd.concat(raw_rows, ignore_index=True), pd.DataFrame(summary_rows))


def load_zebrafish() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[load] zebrafish expression", flush=True)
    expr = pd.read_csv(ZEBRAFISH_EXPR, index_col=0)
    meta = pd.read_csv(ZEBRAFISH_META, index_col=0).reindex(expr.index)
    groups = pd.read_csv(ZEBRAFISH_GROUPS, index_col=0).reindex(expr.index)
    seg = pd.read_csv(ZEBRAFISH_SEGMENTS)
    segment_to_lineage = dict(zip(seg.segment.astype(float), seg.lineage.astype(str)))
    lineage = pd.to_numeric(groups.segment, errors="coerce").map(segment_to_lineage)
    if meta.HPF.isna().any() or groups.index.hasnans:
        raise ValueError("Zebrafish expression contains barcodes absent from metadata")

    genes = [c for c in expr.columns if c != "time"]
    modules = pd.read_csv(ZEBRAFISH_MODULES)
    modules = modules[modules.score_eligible.astype(bool)].copy()
    modules["gene"] = modules.gene.astype(str).str.upper()
    module_sets = modules.groupby("lineage").gene.apply(lambda x: sorted(set(x))).to_dict()

    rank_rows: list[pd.DataFrame] = []
    all_ranked: dict[tuple[str, int, str, float, str], pd.DataFrame] = {}
    for seed in range(3):
        for condition, prefix in [("genuine GRN", "real"), ("no GRN", "nogrn")]:
            for readout in ["drift", "expression"]:
                data = pd.read_csv(ZEBRAFISH_GENERATED / f"{prefix}_s{seed}.{readout}.csv")
                for (cell_type, stage_hpf), unit in data.groupby(["cell_type", "stage_hpf"]):
                    if float(stage_hpf) not in ZEBRAFISH_PRIMARY_HPF:
                        continue
                    averaged = unit.groupby("gene", as_index=False)[readout].mean()
                    ranked = rank_frame(averaged, readout)
                    ranked.insert(0, "condition", condition)
                    ranked.insert(0, "stage", float(stage_hpf))
                    ranked.insert(0, "lineage", cell_type)
                    ranked.insert(0, "dataset", "zebrafish")
                    ranked.insert(4, "seed", seed)
                    ranked.insert(5, "readout", readout)
                    ranked = ranked.rename(columns={readout: "value"})
                    rank_rows.append(ranked)
                    all_ranked[(readout, seed, condition, float(stage_hpf), cell_type)] = ranked.set_index("gene")

    screen_rows: list[dict] = []
    for cell_type, markers in module_sets.items():
        for hpf in sorted(ZEBRAFISH_PRIMARY_HPF):
            mask = lineage.eq(cell_type) & meta.HPF.eq(hpf)
            if not mask.any():
                continue
            x = expr.loc[mask, genes]
            means = x.mean(axis=0)
            detect = x.gt(0).mean(axis=0)
            mean_pct = means.rank(method="average", pct=True)
            detect_pct = detect.rank(method="average", pct=True)
            for gene in markers:
                if gene not in means.index:
                    continue
                seed_rows = []
                for seed in range(3):
                    grn = all_ranked.get(("drift", seed, "genuine GRN", hpf, cell_type))
                    nogrn = all_ranked.get(("drift", seed, "no GRN", hpf, cell_type))
                    if grn is None or nogrn is None or gene not in grn.index or gene not in nogrn.index:
                        continue
                    seed_rows.append((seed, grn.loc[gene], nogrn.loc[gene]))
                if not seed_rows:
                    continue
                for seed, grn, nogrn in seed_rows:
                    screen_rows.append({
                        "dataset": "zebrafish", "gene": gene, "lineage": cell_type,
                        "lineage_display": cell_type.replace("_", " "), "time": hpf,
                        "heldout": True, "n_cells": int(mask.sum()),
                        "observed_mean": means[gene], "detection_fraction": detect[gene],
                        "expression_percentile": mean_pct[gene],
                        "detection_percentile": detect_pct[gene],
                        "grn_rank": int(grn["rank"]), "nogrn_rank": int(nogrn["rank"]),
                        "grn_rank_score": grn["rank_score"],
                        "nogrn_rank_score": nogrn["rank_score"],
                        "rank_gain": int(nogrn["rank"] - grn["rank"]),
                        "rank_score_gain": grn["rank_score"] - nogrn["rank_score"],
                        "seed": seed, "reference_panel": "Farrell frozen modules",
                    })

    raw_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for hpf in sorted(meta.HPF.unique()):
        mask = lineage.eq("adaxial_cells") & meta.HPF.eq(hpf)
        values = expr.loc[mask, "MYOD1"]
        summary_rows.append({"dataset": "zebrafish", "gene": "MYOD1",
                             "lineage": "adaxial_cells", "time": hpf,
                             "heldout": hpf in HELDOUT_ZEBRAFISH, **distribution_summary(values)})
        raw_rows.append(pd.DataFrame({
            "dataset": "zebrafish", "gene": "MYOD1", "lineage": "adaxial_cells",
            "time": hpf, "heldout": hpf in HELDOUT_ZEBRAFISH,
            "cell": values.index.astype(str), "expression": values.to_numpy(),
        }))
    return (pd.DataFrame(screen_rows), pd.concat(rank_rows, ignore_index=True),
            pd.concat(raw_rows, ignore_index=True), pd.DataFrame(summary_rows))


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0, "axes.titlesize": 9.2,
        "axes.labelsize": 8.2, "xtick.labelsize": 7.1, "ytick.labelsize": 7.1,
        "axes.linewidth": 0.65, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": COLORS["paper"], "figure.facecolor": COLORS["paper"],
        "axes.facecolor": COLORS["paper"],
    })


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", va="top", ha="left", color="#15191C")


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.85, zorder=0)


def add_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.985, title, fontsize=15.0, fontweight="bold", va="top", color="#111619")
    fig.text(0.055, 0.956, subtitle, fontsize=8.7, va="top", color="#586168")


def add_violin_trajectory(ax: plt.Axes, raw: pd.DataFrame, summary: pd.DataFrame,
                          times: list[float], heldout: set[float], color: str) -> None:
    for t in times:
        if t in heldout:
            ax.axvspan(t - 0.32, t + 0.32, color=COLORS["heldout"], alpha=0.55, zorder=0)
    positions, vals = [], []
    for t in times:
        v = raw.loc[raw.time.eq(t), "expression"].dropna().to_numpy()
        if len(v) >= 2:
            positions.append(t); vals.append(v)
        elif len(v) == 1:
            ax.scatter([t], v, s=12, color=color, zorder=4)
    if vals:
        vp = ax.violinplot(vals, positions=positions, widths=0.52, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.18)
    use = summary[summary.n_cells.gt(0)].sort_values("time")
    ax.plot(use.time, use.observed_median, color=color, linewidth=1.45, zorder=4)
    ax.vlines(use.time, use.observed_q25, use.observed_q75, color=color, linewidth=1.15, zorder=4)
    ax.scatter(use.time, use.observed_median, s=17, facecolor=COLORS["paper"],
               edgecolor=color, linewidth=0.9, zorder=5)
    for row in use.itertuples():
        ax.text(row.time, ax.get_ylim()[0], str(int(row.n_cells)), ha="center", va="bottom",
                fontsize=5.8, color="#7A8388", clip_on=False)
    clean_axis(ax)


def draw_rank_compass(ax: plt.Axes, rows: pd.DataFrame, topk: int, seeds: bool = False) -> None:
    ax.axvline(topk, color="#AEB5B8", linestyle=(0, (2, 2)), linewidth=0.8, zorder=0)
    if seeds:
        for seed, g in rows.groupby("seed"):
            by = g.set_index("condition")
            if not {"no GRN", "genuine GRN"}.issubset(by.index):
                continue
            x0, x1 = by.loc["no GRN", "rank"], by.loc["genuine GRN", "rank"]
            ax.plot([x0, x1], [seed, seed], color="#A6ADB0", linewidth=0.9, zorder=1)
            ax.scatter(x0, seed, s=22, facecolor=COLORS["paper"], edgecolor=COLORS["nogrn"], zorder=3)
            ax.scatter(x1, seed, s=24, color=COLORS["grn"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_yticks([0, 1, 2], ["seed 0", "seed 1", "seed 2"])
    else:
        by = rows.set_index("condition")
        x0, x1 = by.loc["no GRN", "rank"], by.loc["genuine GRN", "rank"]
        ax.plot([x0, x1], [0, 0], color="#A6ADB0", linewidth=1.2)
        ax.scatter(x0, 0, s=34, facecolor=COLORS["paper"], edgecolor=COLORS["nogrn"], zorder=3)
        ax.scatter(x1, 0, s=36, color=COLORS["grn"], edgecolor="white", linewidth=0.4, zorder=3)
        ax.text(x0, 0.13, f"no GRN  {int(x0)}", ha="center", fontsize=6.6, color=COLORS["nogrn"])
        ax.text(x1, -0.18, f"GRN  {int(x1)}", ha="center", va="top", fontsize=6.6, color=COLORS["grn"])
        ax.set_yticks([])
        ax.set_ylim(-0.42, 0.42)
    ax.set_xscale("log")
    ax.set_xlabel("decoded-drift rank  (rank 1 is best)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5)


def aggregate_screen(screen: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "gene", "lineage", "lineage_display", "time", "heldout",
            "n_cells", "observed_mean", "detection_fraction", "expression_percentile",
            "detection_percentile", "reference_panel"]
    return (screen.groupby(keys, dropna=False, as_index=False)
            .agg(grn_rank=("grn_rank", "median"), nogrn_rank=("nogrn_rank", "median"),
                 grn_rank_score=("grn_rank_score", "median"),
                 nogrn_rank_score=("nogrn_rank_score", "median"),
                 rank_gain=("rank_gain", "median"), rank_score_gain=("rank_score_gain", "median"),
                 n_seeds=("seed", lambda x: int(x.notna().sum()))))


def plot_systematic_screen(ax: plt.Axes, data: pd.DataFrame) -> None:
    for dataset, color, marker in [("pancreas", COLORS["pancreas"], "o"),
                                   ("zebrafish", COLORS["zebrafish"], "s")]:
        d = data[data.dataset.eq(dataset)]
        ax.scatter(d.expression_percentile, d.rank_score_gain, s=17, marker=marker,
                   color=color, alpha=0.42, edgecolor="none", label=dataset, zorder=2)
        bins = pd.cut(d.expression_percentile, np.linspace(0, 1, 6), include_lowest=True)
        med = d.groupby(bins, observed=True).agg(
            x=("expression_percentile", "median"), y=("rank_score_gain", "median")).dropna()
        ax.plot(med.x, med.y, color=color, linewidth=2.0, marker=marker, markersize=4.2, zorder=4)
    ax.axhline(0, color="#7E878C", linewidth=0.8)
    labels = {
        ("pancreas", "NEUROG3", "neurog3_late", 3.0),
        ("pancreas", "SLC30A8", "sc_beta", 6.0),
        ("zebrafish", "MYOD1", "adaxial_cells", 9.0),
    }
    for row in data.itertuples():
        if (row.dataset, row.gene, row.lineage, float(row.time)) not in labels:
            continue
        dx, dy = (0.018, 0.025 if row.gene != "SLC30A8" else -0.05)
        ax.annotate(row.gene, (row.expression_percentile, row.rank_score_gain),
                    xytext=(row.expression_percentile + dx, row.rank_score_gain + dy),
                    arrowprops=dict(arrowstyle="-", color="#7B8489", lw=0.6),
                    fontsize=7.2, fontstyle="italic", color="#202629")
    ax.set(xlim=(0, 1.03), xlabel="observed expression percentile within lineage and time",
           ylabel="GRN promotion in drift rank\n(fraction of ranked gene list)")
    ax.set_title("Where regulatory guidance adds value beyond the VAE baseline", loc="left", pad=5)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.4, columnspacing=1.0)


def survival_curve(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, 101)
    return x, np.asarray([(scores >= threshold).mean() for threshold in x])


def plot_reference_survival(ax: plt.Axes, rank_long: pd.DataFrame,
                            screen_agg: pd.DataFrame) -> None:
    # Use the exact frozen gene-unit universe from the screen; zebrafish seeds are averaged first.
    primary = screen_agg[~screen_agg.reference_panel.str.contains("boundary control", na=False)]
    rows = []
    for condition, col in [("no GRN", "nogrn_rank_score"), ("genuine GRN", "grn_rank_score")]:
        for dataset in ["pancreas", "zebrafish"]:
            scores = primary.loc[primary.dataset.eq(dataset), col].dropna().to_numpy()
            x, y = survival_curve(scores)
            rows.append(pd.DataFrame({"dataset": dataset, "condition": condition,
                                      "threshold": x, "fraction_at_or_above": y}))
            ls = "-" if condition == "genuine GRN" else (0, (3, 2))
            color = COLORS["pancreas"] if dataset == "pancreas" else COLORS["zebrafish"]
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.75,
                    alpha=1.0 if condition == "genuine GRN" else 0.72)
    ax.set(xlabel="drift-rank score  (1 = top-ranked)",
           ylabel="fraction of frozen reference genes\nat or above threshold", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Reference programs shift upward without requiring every gene to move", loc="left", pad=5)
    clean_axis(ax)
    handles = [
        Line2D([0], [0], color=COLORS["pancreas"], lw=2, label="pancreas"),
        Line2D([0], [0], color=COLORS["zebrafish"], lw=2, label="zebrafish"),
        Line2D([0], [0], color="#555", lw=1.7, label="genuine GRN"),
        Line2D([0], [0], color="#777", lw=1.7, linestyle=(0, (3, 2)), label="no GRN"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), fontsize=7)
    return pd.concat(rows, ignore_index=True)


def plot_figure2(screen: pd.DataFrame, rank_long: pd.DataFrame, raw: pd.DataFrame,
                 summary: pd.DataFrame, out_dir: Path, dpi: int) -> pd.DataFrame:
    set_style()
    agg = aggregate_screen(screen)
    fig = plt.figure(figsize=(10.8, 10.5))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.5, 0.85],
                     hspace=0.53, wspace=0.26, top=0.91, bottom=0.08, left=0.08, right=0.97)
    ax_a = fig.add_subplot(outer[0, :])
    plot_systematic_screen(ax_a, agg); panel_label(ax_a, "a")

    for col, dataset, gene, lineage, times, heldout, color, topk in [
        (0, "pancreas", "NEUROG3", "neurog3_late", list(range(8)), HELDOUT_PANCREAS,
         COLORS["pancreas"], 15),
        (1, "zebrafish", "MYOD1", "adaxial_cells",
         sorted(summary.loc[summary.dataset.eq("zebrafish"), "time"].unique()),
         HELDOUT_ZEBRAFISH, COLORS["zebrafish"], 50),
    ]:
        sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, col],
                                      height_ratios=[3.0, 1.0], hspace=0.28)
        ax_top = fig.add_subplot(sub[0]); ax_bottom = fig.add_subplot(sub[1])
        r = raw[(raw.dataset == dataset) & (raw.gene == gene) & (raw.lineage == lineage)]
        s = summary[(summary.dataset == dataset) & (summary.gene == gene) &
                    (summary.lineage == lineage)]
        add_violin_trajectory(ax_top, r, s, times, heldout, color)
        ax_top.set_ylabel("observed normalized expression")
        ax_top.set_xticks(times)
        if dataset == "zebrafish":
            ax_top.set_xticklabels([f"{x:g}" for x in times], rotation=45, ha="right")
            ax_top.set_xlabel("hours post-fertilization   (numbers at baseline: assigned cells)")
            title = r"$\it{MYOD1}$ in adaxial cells"
        else:
            ax_top.set_xlabel("differentiation day   (numbers at baseline: cells)")
            title = r"$\it{NEUROG3}$ in late endocrine progenitors"
        ax_top.set_title(title, loc="left", pad=5)
        if dataset == "pancreas":
            rr = rank_long[(rank_long.dataset == dataset) & (rank_long.lineage == lineage) &
                           (rank_long.stage == 3) & (rank_long.gene == gene)]
            draw_rank_compass(ax_bottom, rr, topk, seeds=False)
            panel_label(ax_top, "b")
        else:
            rr = rank_long[(rank_long.dataset == dataset) & (rank_long.readout == "drift") &
                           (rank_long.lineage == lineage) & (rank_long.stage == 9) &
                           (rank_long.gene == gene)]
            draw_rank_compass(ax_bottom, rr, topk, seeds=True)
            panel_label(ax_top, "c")

    ax_d = fig.add_subplot(outer[2, :])
    surv = plot_reference_survival(ax_d, rank_long, agg); panel_label(ax_d, "d")
    add_title(fig, "GRN guidance adds regulatory attribution beyond VAE-extracted patterns",
              "Observed expression provides context; decoded drift rank measures the additional local regulatory readout.")
    save_figure(fig, out_dir / "figure2_drift_expression_context", dpi)
    return surv


def load_pancreas_trajectories() -> pd.DataFrame:
    rows = []
    for condition, path, time_col in [
        ("observed", PANCREAS_OBS_TRAJ, "ti"),
        ("genuine GRN", PANCREAS_GRN_TRAJ, "t"),
        ("no GRN", PANCREAS_NOGRN_TRAJ, "t"),
    ]:
        d = pd.read_csv(path).rename(columns={time_col: "time"})
        rows.append(d.melt(id_vars="time", var_name="gene", value_name="expression")
                    .assign(dataset="pancreas", condition=condition,
                            provenance="observed population mean" if condition == "observed"
                            else "matched 40-epoch mechanistic diagnostic"))
    return pd.concat(rows, ignore_index=True)


def pancreas_local_integrated(rank_long: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    unit_map = {
        "NEUROG3": ("neurog3_late", 3), "NEUROD1": ("neurog3_late", 3),
        "FEV": ("neurog3_late", 3), "INS": ("sc_beta", 6),
        "GCG": ("sc_alpha", 6), "SST": ("sst_hhex", 6),
        "SLC30A8": ("sc_beta", 6),
    }
    rows = []
    for gene, (lineage, time) in unit_map.items():
        unit = rank_long[(rank_long.dataset == "pancreas") & (rank_long.gene == gene) &
                         (rank_long.lineage == lineage) & (rank_long.stage == time)]
        by = unit.set_index("condition")
        if not {"genuine GRN", "no GRN"}.issubset(by.index):
            continue
        obs = traj[(traj.gene == gene) & (traj.condition == "observed")].set_index("time").expression
        grn = traj[(traj.gene == gene) & (traj.condition == "genuine GRN")].set_index("time").expression
        nogrn = traj[(traj.gene == gene) & (traj.condition == "no GRN")].set_index("time").expression
        common = obs.index.intersection(grn.index).intersection(nogrn.index)
        mae_grn = float((obs.loc[common] - grn.loc[common]).abs().mean())
        mae_nogrn = float((obs.loc[common] - nogrn.loc[common]).abs().mean())
        rows.append({"dataset": "pancreas", "gene": gene, "lineage": lineage, "time": time,
                     "local_rank_score_gain": float(by.loc["genuine GRN", "rank_score"] -
                                                    by.loc["no GRN", "rank_score"]),
                     "integrated_gain": mae_nogrn - mae_grn,
                     "integrated_metric": "MAE(no GRN) - MAE(genuine GRN)",
                     "grn_mae": mae_grn, "nogrn_mae": mae_nogrn, "n_seeds": 0})
    return pd.DataFrame(rows)


def zebrafish_local_integrated(screen: pd.DataFrame, rank_long: pd.DataFrame) -> pd.DataFrame:
    drift = screen.copy()
    expr = rank_long[(rank_long.dataset == "zebrafish") & (rank_long.readout == "expression")]
    expr_wide = (expr.pivot_table(index=["seed", "lineage", "stage", "gene"],
                                  columns="condition", values="rank_score", aggfunc="first")
                 .reset_index())
    expr_wide["integrated_gain"] = expr_wide["genuine GRN"] - expr_wide["no GRN"]
    merged = drift.merge(expr_wide[["seed", "lineage", "stage", "gene", "integrated_gain"]],
                         left_on=["seed", "lineage", "time", "gene"],
                         right_on=["seed", "lineage", "stage", "gene"], how="inner")
    return (merged.groupby(["dataset", "gene", "lineage", "time"], as_index=False)
            .agg(local_rank_score_gain=("rank_score_gain", "median"),
                 integrated_gain=("integrated_gain", "median"),
                 n_seeds=("seed", "nunique"))
            .assign(integrated_metric="generated-expression rank-score gain"))


def plot_trajectory_small_multiples(fig: plt.Figure, spec, traj: pd.DataFrame) -> list[plt.Axes]:
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.24)
    axes = []
    for i, gene in enumerate(["NEUROG3", "NEUROD1", "SLC30A8"]):
        ax = fig.add_subplot(sub[i]); axes.append(ax)
        for day in HELDOUT_PANCREAS:
            ax.axvspan(day - 0.18, day + 0.18, color=COLORS["heldout"], alpha=0.62, zorder=0)
        for condition, color, ls, marker in [
            ("observed", COLORS["observed"], "-", "o"),
            ("no GRN", COLORS["nogrn"], (0, (3, 2)), "s"),
            ("genuine GRN", COLORS["grn"], "-", "^"),
        ]:
            d = traj[(traj.gene == gene) & (traj.condition == condition)]
            ax.plot(d.time, d.expression, color=color, linestyle=ls, marker=marker,
                    markersize=3.2, linewidth=1.35, label=condition, zorder=3)
        ax.set_title(rf"$\it{{{gene}}}$", loc="left")
        ax.set_xlabel("day")
        if i == 0:
            ax.set_ylabel("mean normalized expression")
        clean_axis(ax)
    axes[0].legend(frameon=False, fontsize=6.7, loc="upper right")
    axes[0].text(0.0, 1.16, "Matched 40-epoch mechanistic diagnostic", transform=axes[0].transAxes,
                 fontsize=8.8, fontweight="bold", ha="left")
    return axes


def plot_effect_map(ax: plt.Axes, data: pd.DataFrame, dataset: str) -> None:
    d = data[data.dataset.eq(dataset)]
    color = COLORS[dataset]
    ax.scatter(d.local_rank_score_gain, d.integrated_gain, s=16 if dataset == "zebrafish" else 27,
               color=color, alpha=0.28 if dataset == "zebrafish" else 0.72,
               edgecolor="none", zorder=2)
    ax.axhline(0, color="#7E878C", linewidth=0.75)
    ax.axvline(0, color="#7E878C", linewidth=0.75)
    label_genes = {"NEUROG3", "SLC30A8"} if dataset == "pancreas" else {"MYOD1", "PCDH8"}
    labels = d[d.gene.isin(label_genes)].copy()
    if dataset == "zebrafish":
        labels = labels[((labels.gene == "MYOD1") & (labels.lineage == "adaxial_cells") &
                         (labels.time == 9)) |
                        ((labels.gene == "PCDH8") & (labels.lineage == "somites") &
                         (labels.time == 9))]
    for row in labels.itertuples():
        offset = (-38, 8) if row.gene == "PCDH8" else (5, 10)
        ax.annotate(row.gene, (row.local_rank_score_gain, row.integrated_gain),
                    xytext=offset, textcoords="offset points", fontsize=6.8,
                    fontstyle="italic")
    ax.set_xlabel("local decoded-drift rank-score gain")
    if dataset == "pancreas":
        ax.set_ylabel("trajectory gain\nMAE(no GRN) − MAE(GRN)")
        ax.set_title("Pancreas: local rank vs trajectory error", loc="left", fontsize=8.5)
    else:
        ax.set_ylabel("generated-expression rank-score gain")
        ax.set_title("Zebrafish: local rank vs generated-expression rank", loc="left", fontsize=8.5)
    clean_axis(ax)
    ax.text(0.98, 0.04, "local gain only", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.2, color="#717A7F")


def exemplar_flow(rank_long: pd.DataFrame) -> pd.DataFrame:
    return rank_long[(rank_long.dataset == "zebrafish") & (rank_long.stage == 9) &
                     (((rank_long.lineage == "adaxial_cells") & (rank_long.gene == "MYOD1")) |
                      ((rank_long.lineage == "somites") & (rank_long.gene == "PCDH8")))].copy()


def plot_exemplar_flow(ax: plt.Axes, flow: pd.DataFrame) -> None:
    xpos = {("drift", "no GRN"): 0, ("drift", "genuine GRN"): 1,
            ("expression", "no GRN"): 2.2, ("expression", "genuine GRN"): 3.2}
    seed_colors = ["#4B7CA7", "#9A6BAF", "#D28D42"]
    offsets = {"MYOD1": -0.06, "PCDH8": 0.06}
    for gene in ["MYOD1", "PCDH8"]:
        gd = flow[flow.gene.eq(gene)]
        for seed in range(3):
            for readout in ["drift", "expression"]:
                q = gd[(gd.seed == seed) & (gd.readout == readout)].set_index("condition")
                if len(q) != 2:
                    continue
                x0, x1 = xpos[(readout, "no GRN")], xpos[(readout, "genuine GRN")]
                y0, y1 = q.loc["no GRN", "rank"], q.loc["genuine GRN", "rank"]
                ax.plot([x0, x1], [y0, y1], color=seed_colors[seed], linewidth=0.85, alpha=0.9)
                ax.scatter([x0], [y0], s=15, facecolor=COLORS["paper"], edgecolor=seed_colors[seed], zorder=3)
                ax.scatter([x1], [y1], s=16, color=seed_colors[seed], edgecolor="white", linewidth=0.3, zorder=3)
        med = gd.groupby(["readout", "condition"], as_index=False)["rank"].median()
        for row in med.itertuples():
            ax.text(xpos[(row.readout, row.condition)] + offsets[gene], row.rank,
                    gene if row.condition == "genuine GRN" else "", fontsize=6.3,
                    fontstyle="italic", ha="left", va="center")
    ax.set_yscale("log"); ax.invert_yaxis()
    ax.set_xticks([0, 1, 2.2, 3.2], ["no GRN", "GRN", "no GRN", "GRN"])
    ax.text(0.5, 1.015, "decoded drift", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.2)
    ax.text(2.7, 1.015, "generated expression", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.2)
    ax.set_ylabel("rank  (log scale; rank 1 at top)")
    clean_axis(ax)
    handles = [Line2D([0], [0], color=c, marker="o", lw=1, label=f"seed {i}")
               for i, c in enumerate(seed_colors)]
    ax.legend(handles=handles, frameon=False, fontsize=6.4, ncol=3, loc="lower center")


def plot_summary_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.02, COLORS["nogrn"], "VAE pattern extraction", "Expression geometry can\nalready recover loud identity\nand terminal-state signals.\nExample: SLC30A8."),
        (0.52, COLORS["grn"], "+ regulatory guidance", "Adds lineage-appropriate\nlocal attribution for\ncommitment regulators.\nExample: NEUROG3 / MYOD1."),
    ]
    for x, color, title, body in boxes:
        patch = FancyBboxPatch((x, 0.18), 0.44, 0.67, boxstyle="round,pad=0.018,rounding_size=0.02",
                               facecolor="#FFFFFF", edgecolor=color, linewidth=1.25)
        ax.add_patch(patch)
        ax.text(x + 0.025, 0.72, title, fontsize=8.3, fontweight="bold", color=color)
        ax.text(x + 0.025, 0.57, body, fontsize=6.7, color="#40484D", va="top", linespacing=1.28)
    ax.annotate("distinct evaluation axis", xy=(0.75, 0.09), xytext=(0.25, 0.09),
                arrowprops=dict(arrowstyle="<->", color="#7C858A", lw=0.9),
                ha="center", va="center", fontsize=6.7, color="#687177")


def plot_figure3(traj: pd.DataFrame, effects: pd.DataFrame, flow: pd.DataFrame,
                 out_dir: Path, dpi: int) -> None:
    set_style()
    fig = plt.figure(figsize=(10.8, 10.1))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 0.78],
                     hspace=0.68, wspace=0.31, top=0.89, bottom=0.07, left=0.08, right=0.97)
    axes_a = plot_trajectory_small_multiples(fig, outer[0, :], traj)
    panel_label(axes_a[0], "a")
    ax_b1 = fig.add_subplot(outer[1, 0]); ax_b2 = fig.add_subplot(outer[1, 1])
    plot_effect_map(ax_b1, effects, "pancreas"); plot_effect_map(ax_b2, effects, "zebrafish")
    panel_label(ax_b1, "b")
    ax_c = fig.add_subplot(outer[2, 0]); plot_exemplar_flow(ax_c, flow); panel_label(ax_c, "c")
    ax_c.set_title("Zebrafish exemplar flow at 9 hpf", loc="left", fontsize=8.4, pad=18)
    ax_d = fig.add_subplot(outer[2, 1]); plot_summary_schematic(ax_d); panel_label(ax_d, "d")
    ax_d.set_title("Two useful lenses, two separate tests", loc="left", fontsize=8.5, pad=5)
    add_title(fig, "Local regulatory attribution and integrated expression transport are distinct",
              "A VAE-informed baseline can model expression patterns while GRN guidance selectively reorganizes the local vector field.")
    save_figure(fig, out_dir / "figure3_local_vs_integrated", dpi)


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    for suffix in [".pdf", ".svg", ".png"]:
        fig.savefig(stem.with_suffix(suffix), dpi=dpi if suffix == ".png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {stem}.{{pdf,svg,png}}", flush=True)


def main() -> None:
    args = parse_args()
    args.figure_out_dir.mkdir(parents=True, exist_ok=True)
    args.data_out_dir.mkdir(parents=True, exist_ok=True)

    p_screen, p_rank, p_raw, p_summary = load_pancreas()
    z_screen, z_rank, z_raw, z_summary = load_zebrafish()
    screen = pd.concat([p_screen, z_screen], ignore_index=True)
    rank_long = pd.concat([p_rank.assign(readout="drift"), z_rank], ignore_index=True)
    raw = pd.concat([p_raw, z_raw], ignore_index=True)
    summary = pd.concat([p_summary, z_summary], ignore_index=True)

    screen_agg = aggregate_screen(screen)
    survival = plot_figure2(screen, rank_long, raw, summary, args.figure_out_dir, args.dpi)
    traj = load_pancreas_trajectories()
    effects = pd.concat([pancreas_local_integrated(p_rank, traj),
                         zebrafish_local_integrated(z_screen, z_rank)], ignore_index=True)
    flow = exemplar_flow(z_rank)
    plot_figure3(traj, effects, flow, args.figure_out_dir, args.dpi)

    exports = {
        "figure2_reference_screen_seed_level.csv": screen,
        "figure2_reference_screen_aggregated.csv": screen_agg,
        "figure2_reference_rank_survival.csv": survival,
        "figure2_exemplar_expression_raw.csv": raw,
        "figure2_exemplar_expression_summary.csv": summary,
        "figure3_pancreas_trajectories.csv": traj,
        "figure3_local_vs_integrated.csv": effects,
        "figure3_zebrafish_exemplar_flow.csv": flow,
    }
    for name, data in exports.items():
        data.to_csv(args.data_out_dir / name, index=False)
        print(f"[write] {name}: {len(data):,} rows", flush=True)

    inputs = [PANCREAS_EXPR, PANCREAS_META, PANCREAS_DRIFT, PANCREAS_OBS_TRAJ,
              PANCREAS_GRN_TRAJ, PANCREAS_NOGRN_TRAJ, ZEBRAFISH_EXPR, ZEBRAFISH_META,
              ZEBRAFISH_GROUPS, ZEBRAFISH_SEGMENTS, ZEBRAFISH_MODULES]
    inputs += sorted(ZEBRAFISH_GENERATED.glob("real_s[0-2].drift.csv"))
    inputs += sorted(ZEBRAFISH_GENERATED.glob("real_s[0-2].expression.csv"))
    inputs += sorted(ZEBRAFISH_GENERATED.glob("nogrn_s[0-2].drift.csv"))
    inputs += sorted(ZEBRAFISH_GENERATED.glob("nogrn_s[0-2].expression.csv"))
    manifest = {
        "script": str(Path(__file__).resolve()),
        "claims_guardrail": {
            "no_grn_interpretation": "VAE-informed expression-pattern baseline",
            "grn_interpretation": "additional local regulatory attribution",
            "causal_gene_impact_claimed": False,
            "generated_expression_labeled_observed": False,
            "pancreatic_generated_trajectory_scope": "matched 40-epoch mechanistic diagnostic",
        },
        "heldout": {"pancreas_days": sorted(HELDOUT_PANCREAS),
                    "zebrafish_hpf": sorted(HELDOUT_ZEBRAFISH),
                    "zebrafish_primary_hpf": sorted(ZEBRAFISH_PRIMARY_HPF)},
        "ranking": "descending value; gene-symbol ascending tie break; simulation draws averaged within seed",
        "runtime": {"python": sys.version.split()[0], "executable": sys.executable,
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "matplotlib": mpl.__version__},
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "outputs": list(exports) + [
            "figure2_drift_expression_context.pdf", "figure2_drift_expression_context.svg",
            "figure2_drift_expression_context.png", "figure3_local_vs_integrated.pdf",
            "figure3_local_vs_integrated.svg", "figure3_local_vs_integrated.png"],
    }
    manifest["figure_output_directory"] = str(args.figure_out_dir)
    manifest["data_output_directory"] = str(args.data_out_dir)
    (args.data_out_dir / "drift_expression_context_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("[done] figures and exact plotted data complete", flush=True)


if __name__ == "__main__":
    main()
