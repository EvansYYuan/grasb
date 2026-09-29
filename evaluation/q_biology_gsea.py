#!/usr/bin/env python
# =============================================================================
#  Biology meaningfulness (POSITIVE direction) — are a cell type's positive-drift
#  drivers the genes biology says it should express?
#
#  Markers are positive by nature: a marker is a gene specifically EXPRESSED in a
#  cell type (INS for β, GCG for α). There is no "marker for a gene being off", so
#  this check is one-directional: it asks whether a type's own marker genes are
#  enriched among the genes its drift drives UP. Preranked GSEA (gseapy.prerank)
#  of each type's Veres ED-Table-2 markers against its own drift ranking.
#  The ranking is SIGNED, so the NES is signed too: +NES = the type's own markers
#  are among the genes it drives UP (the expected case); −NES = its own markers are
#  driven DOWN — which can be correct biology, not a miss (e.g. sc_alpha drives GCG
#  down ≈ −1.47 = a transient poly-hormonal population resolving its alpha identity).
#
#  This is the meaningfulness companion to RSS (specificity, expression side).
#  The NEGATIVE direction (specifically-killed genes) has NO marker reference, so
#  it is handled separately by pathway-GSEA / literature, NOT here.
#  See docs/biology_meaningfulness_gsea.md.
# =============================================================================
import argparse, os
import pandas as pd
import gseapy as gp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MARKERS = os.path.join(ROOT, "config", "pancreas", "marker_panel.csv")


def load_marker_sets(path=DEFAULT_MARKERS):
    """Load {marker_set: {lineage: [genes]}} from the versioned panel table."""
    panel = pd.read_csv(path, dtype=str)
    required = {"marker_set", "lineage", "gene"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Marker panel is missing columns: {sorted(missing)}")
    panel = panel.dropna(subset=list(required)).drop_duplicates(
        ["marker_set", "lineage", "gene"]
    )
    return {
        marker_set: group.groupby("lineage", sort=False).gene.apply(list).to_dict()
        for marker_set, group in panel.groupby("marker_set", sort=False)
    }


MARKER_SETS = load_marker_sets()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=os.path.join(ROOT, "results", "per_type_drift.csv"))
    ap.add_argument("--tags", default="grasb,nogrn,shuffle")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--markers", default=DEFAULT_MARKERS)
    ap.add_argument("--marker-set", default="edt2")
    ap.add_argument("--out", default=None,
                    help="default: output/biology_gsea[_<marker-set>].csv")
    args = ap.parse_args()
    tags = args.tags.split(",")
    marker_sets = load_marker_sets(args.markers)
    if args.marker_set not in marker_sets:
        raise ValueError(
            f"Unknown marker set {args.marker_set!r}; choose from {sorted(marker_sets)}"
        )
    MARKERS = marker_sets[args.marker_set]
    if args.out is None:
        suffix = "" if args.marker_set == "v1" else f"_{args.marker_set}"
        args.out = os.path.join(ROOT, "results", f"biology_gsea{suffix}.csv")
    print(f"[marker-set] {args.marker_set} | own positive markers in own drift -> "
          f"{os.path.basename(args.out)}")

    full = pd.read_csv(args.full)
    rows = []
    for tag in tags:
        for (ty, stage), g in full[full.tag == tag].groupby(["cell_type", "stage"]):
            # exo (CHGA- non-endocrine umbrella, no single EDT2 marker panel) and
            # prog_sox2 (week-0 root, zero drift rows) have no marker list here, so
            # they fall through. See memory exo-is-nonendocrine-umbrella.
            if ty not in MARKERS:
                continue
            rnk = g[["gene", "drift"]].sort_values("drift", ascending=False)
            res = gp.prerank(rnk=rnk, gene_sets={"markers": MARKERS[ty]},
                             permutation_num=args.n_perm, min_size=3, max_size=2000,
                             seed=0, threads=4, outdir=None, no_plot=True,
                             verbose=False).res2d.set_index("Term")
            if "markers" not in res.index:
                continue
            row = res.loc["markers"]
            rows.append(dict(tag=tag, cell_type=ty, stage=int(stage),
                             nes=float(row["NES"]), nom_p=float(row["NOM p-val"]),
                             fdr=float(row["FDR q-val"]), lead=row["Lead_genes"]))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"[out] {len(out)} (tag,type,stage) -> {args.out}\n")

    print(f"{'='*72}\nOWN MARKERS in own SIGNED drift  (+NES = up-driven; -NES = driven DOWN, may be correct biology e.g. sc_alpha)\n{'='*72}")
    print("  '% up-activated' = fraction of the n (type,stage) UNITS that pass NES>0 & FDR<.25")
    print("   (a rate over cell types, NOT a count of genes; n is the #units, not #markers)")
    print(f"{'tag':>13} | signed mean NES | % up-activated (FDR<.25) | n")
    for tag in tags:
        s = out[out.tag == tag].dropna(subset=["nes"])
        if not len(s):
            continue
        sig = ((s.nes > 0) & (s.fdr < 0.25)).mean()
        print(f"{tag:>13} |  {s.nes.mean():+.2f}   |      {sig:4.0%}        | {len(s)}")

    print(f"\n{'='*72}\nregulon — per type (own markers in own drift)\n{'='*72}")
    r = out[out.tag == "regulon"].dropna(subset=["nes"]).sort_values("nes", ascending=False)
    for _, x in r.iterrows():
        flag = "*" if x.fdr < 0.25 else " "
        print(f"  {x.cell_type:>16} st{x.stage}  NES={x.nes:+.2f}{flag} FDR={x.fdr:.3f}  lead: {x.lead}")
    print("\nDONE")


if __name__ == "__main__":
    main()
