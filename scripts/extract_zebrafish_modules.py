#!/usr/bin/env python
"""Extract preregistered lineage panels from Farrell et al. 2018 Table S5."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZF_CONFIG = ROOT / "config" / "zebrafish"

# Generic technical/cell-cycle genes are removed by an explicit, reviewable rule. We do not
# select genes by their drift ranking or by differential expression in held-out cells.
GENERIC_EXACT = {
    "ACTB1", "ACTB2", "GAPDH", "MALAT1", "MKI67", "PTMAA", "TMSB4X",
    "H2AFX", "HISTH1L", "YWHAZ", "YWHAQA", "TPX2", "NUSAP1", "CENPF",
    "NASP", "FBXO5", "CEP250", "DYNLL1",
}
GENERIC_PREFIX = ("RPL", "RPS", "HSP90", "HSPA", "HSPB", "SEC61", "SSR")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workbook", type=Path, default=ROOT / "data/farrell_table_s5.xlsx")
    ap.add_argument("--mapping", type=Path, default=ZF_CONFIG / "module_mapping.csv")
    ap.add_argument("--hvg", type=Path, default=ROOT / "data/zebrafish_hvg.csv")
    ap.add_argument("--out", type=Path, default=ZF_CONFIG / "published_modules.csv")
    args = ap.parse_args()

    mapping = pd.read_csv(args.mapping)
    hvg_df = pd.read_csv(args.hvg)
    gene_col = "x" if "x" in hvg_df else hvg_df.columns[-1]
    hvg = set(hvg_df[gene_col].astype(str).str.upper())
    rows = []
    cache = {}
    for rec in mapping.itertuples(index=False):
        if rec.sheet not in cache:
            cache[rec.sheet] = pd.read_excel(args.workbook, sheet_name=rec.sheet)
        table = cache[rec.sheet]
        col = f"Module {rec.module_id}"
        weight_col = f"Weights {rec.module_id}"
        if col not in table or weight_col not in table:
            raise KeyError(f"Missing {col}/{weight_col} in sheet {rec.sheet}")
        for rank, (gene, weight) in enumerate(zip(table[col], table[weight_col]), start=1):
            if pd.isna(gene):
                continue
            gene = str(gene).upper()
            generic = gene in GENERIC_EXACT or gene.startswith(GENERIC_PREFIX)
            rows.append({
                "lineage": rec.lineage, "segment": rec.segment, "gene": gene,
                "panel_role": "published_modules", "source_id": "FARRELL2018_TABLE_S5",
                "source_file": args.workbook.name, "source_sheet": rec.sheet,
                "module_id": int(rec.module_id), "module_rank": rank,
                "module_weight": float(weight), "mapping_confidence": rec.confidence,
                "independence": "dataset_derived", "in_hvg": gene in hvg,
                "generic_excluded": generic,
                "score_eligible": bool(gene in hvg and not generic and rec.mapping_status == "frozen"),
            })
    out = pd.DataFrame(rows).sort_values(["lineage", "module_id", "module_rank"])
    # A gene occurring in two selected modules for one lineage remains one marker; retain its
    # highest-weight occurrence and keep the full extraction separately for audit.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out.with_name("published_modules_full.csv"), index=False)
    panel = (out[out.score_eligible].sort_values("module_weight", ascending=False)
             .drop_duplicates(["lineage", "gene"])
             .sort_values(["lineage", "module_id", "module_rank"]))
    lineage_count = panel.groupby("gene").lineage.transform("nunique")
    panel["cross_lineage_unique"] = lineage_count.eq(1)
    panel["specificity_eligible"] = panel["score_eligible"] & panel["cross_lineage_unique"]
    panel.to_csv(args.out, index=False)
    summary = panel.groupby("lineage").agg(
        n_markers=("gene", "size"), n_modules=("module_id", "nunique"),
        n_specificity_markers=("specificity_eligible", "sum"),
        min_confidence=("mapping_confidence", lambda x: "moderate" if "moderate" in set(x) else "high"),
    ).reset_index()
    summary.to_csv(args.out.with_name("published_modules_summary.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
