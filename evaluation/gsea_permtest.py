#!/usr/bin/env python
# =============================================================================
#  GSEA specificity as a CALIBRATED test — the GSEA analogue of rss_permtest.py.
#
#  Problem (raised 2026-07-10): a high signed-mean NES (e.g. joint-frozen no-GRN
#  +0.64) was read as "biology". But NES is computed self-only (type T's OWN
#  markers on T's OWN signed drift). With no GRN there is no per-type mechanism,
#  so a high self-NES may just be REPRESENTATION GEOMETRY + a global maturation
#  flow (the decode-JVP through J_dec carries the decoder's local gene geometry;
#  in a trajectory-aligned latent the forward-transport direction already aligns
#  with marker expression). Exactly the confound we fixed for RSS.
#
#  Fix (same medicine): build the CROSS matrix  NES[drift_type T, marker_set M]
#  — score EVERY type's marker panel against T's drift ranking. "Specificity
#  present" == the diagonal (T's OWN panel) tends to be the argmax / top-rank
#  across marker sets for T's ranking. Null = RELABELLING: per stage permute
#  which type-name each marker column claims, holding the NES matrix fixed.
#  Pool per-stage units, Monte-Carlo B times -> z + one-sided p for
#     (1) self=argmax count   (higher = more specific)
#     (2) mean self-rank       (lower  = more specific)
#  Also report the raw diagonal (self) mean NES for reference — the uncalibrated
#  number. If self is NOT above the relabelling null, a high self-NES is geometry,
#  not per-type biology.
#
#  Reuses q_biology_gsea's EDT2 panel + gseapy.prerank so the NES are the same
#  objects, only now with a calibrated cross-type null.
# =============================================================================
import argparse, os, sys
import numpy as np
import pandas as pd
import gseapy as gp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q_biology_gsea import DEFAULT_MARKERS, load_marker_sets


def nes_matrices(full, tag, MARKERS, n_perm):
    """Per stage: (types, self_is_argmax dict, per-row NES rank dict, NES matrix M).
    M[T, Mk] = NES of marker-set Mk against drift-type T's signed ranking. Rows/cols
    restricted to types that (a) have a marker panel and (b) are present as a drift
    cell_type this stage, so the diagonal is comparable (square)."""
    per_stage = []
    ft = full[full["tag"] == tag]
    for stage in sorted(ft["stage"].unique()):
        fs = ft[ft["stage"] == stage]
        drift_types = [t for t in fs["cell_type"].unique() if t in MARKERS]
        if len(drift_types) < 2:
            continue
        # only panels whose type is itself a present drift-type -> square, comparable diag
        panels = {t: MARKERS[t] for t in drift_types}
        nes = {}   # nes[T] = {Mk: NES}
        for T, g in fs.groupby("cell_type"):
            if T not in drift_types:
                continue
            rnk = g[["gene", "drift"]].sort_values("drift", ascending=False)
            try:
                res = gp.prerank(rnk=rnk, gene_sets=panels, permutation_num=n_perm,
                                 min_size=3, max_size=2000, seed=0, threads=4,
                                 outdir=None, no_plot=True, verbose=False).res2d.set_index("Term")
            except Exception:
                continue
            nes[T] = {Mk: float(res.loc[Mk, "NES"]) for Mk in panels if Mk in res.index}
        # keep only drift-types whose OWN panel scored (diagonal defined)
        rows = [T for T in drift_types if T in nes and T in nes[T]]
        if len(rows) < 2:
            continue
        # column set = panels scored for ALL kept rows (so every column is comparable)
        cols = [Mk for Mk in rows if all(Mk in nes[T] for T in rows)]
        if len(cols) < 2:
            continue
        M = pd.DataFrame({T: {Mk: nes[T][Mk] for Mk in cols} for T in rows}).T  # index=T, cols=Mk
        self_argmax = {T: (M.loc[T].idxmax() == T) for T in rows}
        row_rank = {}
        for T in rows:
            order = M.loc[T].sort_values(ascending=False).index.tolist()
            row_rank[T] = {Mk: i + 1 for i, Mk in enumerate(order)}
        per_stage.append((rows, cols, self_argmax, row_rank, M))
    return per_stage


def perm_test(per_stage, B, rng):
    obs_hits, obs_rank_sum, n_units, self_nes_sum = 0, 0.0, 0, 0.0
    stage_data = []
    for rows, cols, self_argmax, row_rank, M in per_stage:
        for T in rows:
            obs_hits += int(self_argmax[T])
            obs_rank_sum += row_rank[T][T]
            self_nes_sum += float(M.loc[T, T])
            n_units += 1
        stage_data.append((rows, cols, row_rank))
    obs_frac = obs_hits / n_units
    obs_meanrank = obs_rank_sum / n_units
    self_mean_nes = self_nes_sum / n_units

    null_hits = np.zeros(B); null_ranksum = np.zeros(B)
    for b in range(B):
        h, rs = 0, 0.0
        for rows, cols, row_rank in stage_data:
            # permute which type-name each marker column claims (bijection over cols)
            perm = rng.permutation(cols)
            claim = dict(zip(cols, perm))          # column Mk -> claimed name
            # a row T now "self"-claims the permuted name of column T (if T is a col)
            for T in rows:
                if T in claim:
                    cname = claim[T]
                    h += int(row_rank[T][cname] == 1)     # is claimed col the argmax of row T?
                    rs += row_rank[T][cname]
                else:
                    rs += len(cols)
        null_hits[b] = h; null_ranksum[b] = rs
    null_frac = null_hits / n_units
    null_meanrank = null_ranksum / n_units

    def z_p(obs, null, side):
        mu, sd = null.mean(), null.std() + 1e-12
        z = (obs - mu) / sd
        p = ((np.sum(null >= obs) if side == "greater" else np.sum(null <= obs)) + 1) / (len(null) + 1)
        return z, p, mu, sd

    zA, pA, muA, sdA = z_p(obs_frac, null_frac, "greater")
    zR, pR, muR, sdR = z_p(obs_meanrank, null_meanrank, "less")
    return dict(n=n_units, self_mean_nes=self_mean_nes,
                obs_frac=obs_frac, null_frac_mu=muA, null_frac_sd=sdA, zA=zA, pA=pA,
                obs_meanrank=obs_meanrank, null_rank_mu=muR, null_rank_sd=sdR, zR=zR, pR=pR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--markers", default=DEFAULT_MARKERS)
    ap.add_argument("--marker-set", default="edt2")
    ap.add_argument("--n-perm", type=int, default=200, help="gseapy internal perms per NES")
    ap.add_argument("--B", type=int, default=5000, help="relabelling perms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    marker_sets = load_marker_sets(args.markers)
    if args.marker_set not in marker_sets:
        raise ValueError(
            f"Unknown marker set {args.marker_set!r}; choose from {sorted(marker_sets)}"
        )
    MARKERS = marker_sets[args.marker_set]
    full = pd.read_csv(args.full)
    rng = np.random.default_rng(0)

    print(f"GSEA specificity — cross-type relabelling permutation (marker-set={args.marker_set}, B={args.B})")
    print("self_NES = raw diagonal mean (uncalibrated). self=argmax/self-rank calibrated by")
    print("permuting which type each marker panel claims. self NOT above null => geometry, not per-type biology.")
    print("-" * 104)
    print(f"{'tag':>16} | {'n':>3} | {'self_NES':>8} | {'argmax obs':>10} {'null':>11} {'z':>6} {'p':>8} | "
          f"{'rank obs':>8} {'null':>6} {'z':>6} {'p':>8}")
    print("-" * 104)
    recs = []
    for tag in args.tags.split(","):
        tag = tag.strip()
        ps = nes_matrices(full, tag, MARKERS, args.n_perm)
        if not ps:
            print(f"{tag:>16} | (no rows)"); continue
        r = perm_test(ps, args.B, rng)
        recs.append(dict(tag=tag, **r))
        print(f"{tag:>16} | {r['n']:>3} | {r['self_mean_nes']:>+8.2f} | {r['obs_frac']:>9.0%} "
              f"{r['null_frac_mu']:>5.0%}±{r['null_frac_sd']:>3.0%} {r['zA']:>6.2f} {r['pA']:>8.4f} | "
              f"{r['obs_meanrank']:>8.2f} {r['null_rank_mu']:>6.2f} {r['zR']:>6.2f} {r['pR']:>8.4f}")
    if args.out and recs:
        pd.DataFrame(recs).to_csv(args.out, index=False)
        print(f"\n[out] -> {args.out}")
    print("GSEA_PERMTEST_DONE")


if __name__ == "__main__":
    main()
