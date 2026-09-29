#!/usr/bin/env python
"""Marker-file-driven GSEA, recall, and cross-lineage self-rank for drift CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZF_CONFIG = ROOT / "config" / "zebrafish"


def gsea(ranking: pd.DataFrame, genes: list[str], n_perm: int, seed: int):
    present = sorted(set(genes) & set(ranking.gene))
    if len(present) < 3:
        return None
    ranking = ranking[["gene", "drift"]].sort_values(
        ["drift", "gene"], ascending=[False, True], kind="mergesort").copy()
    # gseapy otherwise warns that exact score ties may be ordered arbitrarily.
    # Move tied genes by adjacent float values in gene-symbol order; non-tied
    # scores are untouched and the adjustment is below measurement precision.
    for _, idx in ranking.groupby("drift", sort=False).groups.items():
        if len(idx) < 2:
            continue
        ordered = sorted(idx, key=lambda i: ranking.at[i, "gene"])
        value = float(ranking.at[ordered[-1], "drift"])
        ranking.at[ordered[-1], "drift"] = value
        for i in reversed(ordered[:-1]):
            value = np.nextafter(value, np.inf)
            ranking.at[i, "drift"] = value
    res = gp.prerank(rnk=ranking, gene_sets={"panel": present},
                     permutation_num=n_perm, min_size=3, max_size=2000, seed=seed,
                     threads=1, outdir=None, no_plot=True, verbose=False).res2d
    if res.empty:
        return None
    x = res.iloc[0]
    return float(x["NES"]), float(x["NOM p-val"]), float(x["FDR q-val"]), str(x["Lead_genes"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=Path, required=True)
    ap.add_argument("--markers", type=Path, default=ZF_CONFIG / "published_modules.csv")
    ap.add_argument("--classical-markers", type=Path,
                    default=ZF_CONFIG / "marker_panel.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/zebrafish_biology")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    drift = pd.read_csv(args.drift)
    markers = pd.read_csv(args.markers)
    full_sets = markers.groupby("lineage").gene.apply(list).to_dict()
    specific_sets = (markers[markers.specificity_eligible]
                     .groupby("lineage").gene.apply(list).to_dict())
    own_rows, cross_rows, recall_rows = [], [], []
    for (tag, lineage, stage), unit in drift.groupby(["tag", "cell_type", "stage"]):
        if lineage not in full_sets:
            continue
        rank = unit.sort_values("drift", ascending=False).copy()
        rank["gene"] = rank.gene.astype(str).str.upper()
        own = gsea(rank, full_sets[lineage], args.n_perm, args.seed)
        if own:
            own_rows.append(dict(tag=tag, cell_type=lineage, stage=stage,
                                 nes=own[0], nom_p=own[1], fdr=own[2], lead=own[3]))
        scores = {}
        for candidate, genes in specific_sets.items():
            ans = gsea(rank, genes, args.n_perm, args.seed)
            if ans:
                scores[candidate] = ans[0]
        if lineage in scores:
            ordered = sorted(scores, key=scores.get, reverse=True)
            cross_rows.append(dict(tag=tag, cell_type=lineage, stage=stage,
                                   self_nes=scores[lineage], self_rank=ordered.index(lineage)+1,
                                   n_lineages=len(ordered), self_is_top=ordered[0] == lineage))
        present = set(full_sets[lineage]) & set(rank.gene)
        top = set(rank[rank.drift > 0].head(args.topk).gene)
        recall_rows.append(dict(tag=tag, cell_type=lineage, stage=stage, topk=args.topk,
                                recall=len(present & top)/len(present) if present else np.nan,
                                n_markers=len(present)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(own_rows).to_csv(args.out_dir / "own_module_gsea.csv", index=False)
    pd.DataFrame(cross_rows).to_csv(args.out_dir / "cross_lineage_gsea.csv", index=False)
    pd.DataFrame(recall_rows).to_csv(args.out_dir / "marker_recall.csv", index=False)

    # Independent classical-marker sensitivity analysis.  Keep these outputs
    # separate from the preregistered published-module primary analysis.
    classical = pd.read_csv(args.classical_markers)
    classical["gene"] = classical.gene.astype(str).str.upper()
    classical_sets = classical.groupby("lineage").gene.apply(list).to_dict()
    membership = classical.groupby("gene").lineage.nunique()
    classical_specific = {
        lineage: [g for g in genes if membership.get(g, 0) == 1]
        for lineage, genes in classical_sets.items()
    }
    c_own, c_cross, c_recall = [], [], []
    for (tag, lineage, stage), unit in drift.groupby(["tag", "cell_type", "stage"]):
        if lineage not in classical_sets:
            continue
        rank = unit.sort_values("drift", ascending=False).copy()
        rank["gene"] = rank.gene.astype(str).str.upper()
        own = gsea(rank, classical_sets[lineage], args.n_perm, args.seed)
        if own:
            c_own.append(dict(tag=tag, cell_type=lineage, stage=stage,
                              nes=own[0], nom_p=own[1], fdr=own[2], lead=own[3]))
        scores = {}
        for candidate, genes in classical_specific.items():
            ans = gsea(rank, genes, args.n_perm, args.seed)
            if ans:
                scores[candidate] = ans[0]
        if lineage in scores:
            ordered = sorted(scores, key=scores.get, reverse=True)
            c_cross.append(dict(tag=tag, cell_type=lineage, stage=stage,
                                self_nes=scores[lineage],
                                self_rank=ordered.index(lineage) + 1,
                                n_lineages=len(ordered),
                                self_is_top=ordered[0] == lineage))
        present = set(classical_sets[lineage]) & set(rank.gene)
        top = set(rank[rank.drift > 0].head(args.topk).gene)
        c_recall.append(dict(tag=tag, cell_type=lineage, stage=stage,
                             topk=args.topk,
                             recall=len(present & top) / len(present) if present else np.nan,
                             n_markers=len(present)))
    pd.DataFrame(c_own).to_csv(args.out_dir / "classical_own_gsea.csv", index=False)
    pd.DataFrame(c_cross).to_csv(args.out_dir / "classical_cross_lineage_gsea.csv", index=False)
    pd.DataFrame(c_recall).to_csv(args.out_dir / "classical_marker_recall.csv", index=False)
    print({"own_gsea_units": len(own_rows), "cross_lineage_units": len(cross_rows),
           "recall_units": len(recall_rows), "classical_own_units": len(c_own),
           "classical_cross_units": len(c_cross),
           "classical_recall_units": len(c_recall)})


if __name__ == "__main__":
    main()
