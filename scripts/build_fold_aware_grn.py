#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the fold-aware, context-specific GRN artifact used by GRASB.

The SB holds out whole timepoints VAL_TPS=[3,6] and reports held-out W2 there. But the
default builder (build_grn_drift.py) fits the GRN prior per (type, interval=t_n∪t_{n+1})
over ALL cells, so the bins touching t3/t6 -- (2,3),(3,4),(5,6),(6,7) -- are fit USING the
held-out cells, and step_to_interval floor-maps the masked-t3 bridge onto exactly those
bins. => the reported W2 is optimistic by leakage.

This builder implements the fold-aware fix the user proposed: mask t3,t6 and use MERGED
intervals (2->4) and (5->7) that skip the held-out week, matching what the SB actually
bridges. It emits TWO artifacts that share the SAME (merged) interval structure and differ
ONLY in whether the held-out cells enter the beta fit -- so training on each and diffing
held-out W2 isolates the LEAKAGE effect from any interval-structure effect:

  --variant clean : merged bin (2->4) fit on t2 ∪ t4 only  (held-out t3 EXCLUDED) -> honest
  --variant leaky : merged bin (2->4) fit on t2 ∪ t3 ∪ t4  (held-out t3 INCLUDED) -> leaks

Non-merged intervals (0,1),(1,2),(4,5) are identical in both variants (no masked middle).
Everything else (detrend, one_pn adaptive alpha, gamma calibration, sparse edge format) is
the default recipe, imported from build_grn_drift so the artifacts drop straight into
train_grn_usb.py --grn-pkl. The engine (grn_drift.py) maps steps to these merged intervals
via boundary search (interval_hi), added alongside this.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_grn_drift import (
    load_expression, attach_labels, load_base_grn_mask, fit_bin, calibrate_gamma,
    prior_from_selection,
)

TRAIN_TPS = [0, 1, 2, 4, 5, 7]      # must match train_grn_usb.py
VAL_TPS = [3, 6]


def merged_intervals(train_tps):
    """Consecutive TRAIN-timepoint pairs -> intervals that skip masked weeks.
    e.g. [0,1,2,4,5,7] -> [(0,1),(1,2),(2,4),(4,5),(5,7)]; (2,4)&(5,7) span masked t3,t6."""
    ts = sorted(train_tps)
    return list(zip(ts[:-1], ts[1:]))


def build_priors(prior, X, cell_type, timepoint, mask, gene_to_idx, intervals,
                 rare_threshold):
    """Precompute the borrow-strength pooled prior(s) β0, ALWAYS on TRAIN cells only.

    The pool is fit on `np.isin(timepoint, TRAIN_TPS)` so a `clean+prior` artifact stays
    honestly clean: the held-out weeks (VAL_TPS) never enter the prior, exactly as they
    never enter the clean bin fits. Otherwise the prior would silently re-leak t3/t6.

    Returns a callable pick(type_k, interval) -> {target_idx: β0} (or None for prior=none):
      type      : one type across all its train intervals (best truth-recovery, 07-02)
      ivl       : all types at one interval          (reproducible, biased generic)
      global    : all train cells                    (strongest pool, most biased)
      type_ivl  : type-pool if that type has >= rare_threshold train cells, else ivl
                  (07-02 recommendation: type-temporal with ivl fallback for rare types)
    """
    if prior == "none":
        return lambda k, iv: None

    train = np.isin(timepoint, TRAIN_TPS)

    def pool(sel, label):
        print(f"[prior] fit {label:>28}  cells={sel.size:5d}")
        return prior_from_selection(X, timepoint, sel, mask, gene_to_idx)

    prior_type, prior_ivl, prior_global = {}, {}, None
    type_n = {k: int(((cell_type == k) & train).sum()) for k in np.unique(cell_type)}

    if prior in ("type", "type_ivl"):
        for k in np.unique(cell_type):
            if prior == "type_ivl" and type_n[k] < rare_threshold:
                continue   # rare type -> will fall back to ivl, don't waste the fit
            sel = np.where((cell_type == k) & train)[0]
            if sel.size >= 1:
                prior_type[k] = pool(sel, f"type={k}")
    if prior in ("ivl", "type_ivl"):
        for (lo, hi) in intervals:
            keep = [t for t in range(lo, hi + 1) if t in TRAIN_TPS]
            sel = np.where(np.isin(timepoint, keep) & train)[0]
            if sel.size >= 1:
                prior_ivl[(lo, hi)] = pool(sel, f"ivl={(lo, hi)}")
    if prior == "global":
        prior_global = pool(np.where(train)[0], "global")

    def pick(k, iv):
        if prior == "global":
            return prior_global
        if prior == "type":
            return prior_type.get(k)
        if prior == "ivl":
            return prior_ivl.get(iv)
        # type_ivl: type-pool when well-populated, else interval fallback
        if type_n.get(k, 0) >= rare_threshold and k in prior_type:
            return prior_type[k]
        return prior_ivl.get(iv)

    return pick


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed",
                    default=os.path.join(ROOT, "data", "pancreatic_preprocessed.csv"))
    ap.add_argument("--meta",
                    default=os.path.join(ROOT, "data", "pancreatic_metadata.tsv"))
    ap.add_argument("--base-grn",
                    default=os.path.join(ROOT, "data", "human_promoter_base_grn.parquet"))
    ap.add_argument("--variant", choices=["clean", "leaky"], default="clean",
                    help="clean = held-out weeks EXCLUDED from merged-bin fits (honest); "
                         "leaky = held-out weeks INCLUDED (reproduces the current leak, "
                         "with the merged interval structure held fixed).")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--alpha-mode", choices=["fixed", "one_pn"], default="one_pn")
    ap.add_argument("--prior", choices=["none", "type", "ivl", "global", "type_ivl"],
                    default="type_ivl",
                    help="borrow-strength pooled prior baked into the bin fit (Block 2.1 "
                         "Part B). none = shrink-to-zero (plain clean/leaky). Pool is ALWAYS "
                         "fit on train cells only, so clean+prior stays leak-free.")
    ap.add_argument("--prior-rare-threshold", type=int, default=200,
                    help="type_ivl: min train cells for a type to trust its own pool; "
                         "below this, fall back to the interval (cross-type) pool.")
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--component", choices=["full", "regulon"], default="regulon",
                    help="Which field gamma is CALIBRATED on. MUST match the runtime "
                         "--grn-target of train_latent_grn.py, else the units knob is fit "
                         "to a field we discard (we deploy regulon => use --component regulon).")
    ap.add_argument("--train-tps", default="0,1,2,4,5,7",
                    help="TRAIN weeks the GRN may see; also sets the merged intervals it is "
                         "fit on. Sparser => longer intervals (delta-t probe). Leak guard: "
                         "must never contain a VAL week.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global TRAIN_TPS
    TRAIN_TPS = [int(t) for t in args.train_tps.split(",")]
    leak = sorted(set(TRAIN_TPS) & set(VAL_TPS))
    if leak:
        raise SystemExit(f"[leak] --train-tps contains held-out week(s) {leak}")
    if args.prior != "none" and args.variant == "leaky":
        print("[warn] --prior with --variant leaky: the bin fit still sees held-out cells; "
              "the clean+prior artifact should use --variant clean.")

    X, genes, time, barcodes = load_expression(args.preprocessed)
    cell_type, timepoint = attach_labels(barcodes, args.meta)
    mask = load_base_grn_mask(args.base_grn, genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    intervals = merged_intervals(TRAIN_TPS)
    print(f"[load] cells={X.shape[0]} genes={X.shape[1]} variant={args.variant} "
          f"prior={args.prior} alpha_mode={args.alpha_mode} alpha={args.alpha}")
    print(f"[intervals] {intervals}  (masked weeks {VAL_TPS})")

    pick_prior = build_priors(args.prior, X, cell_type, timepoint, mask, gene_to_idx,
                              intervals, args.prior_rare_threshold)

    artifacts = {}
    for k in np.unique(cell_type):
        for (lo, hi) in intervals:
            masked_mid = [t for t in range(lo + 1, hi) if t in VAL_TPS]   # weeks skipped by merge
            if args.variant == "clean":
                keep_tps = [lo, hi]                       # endpoints only -> no held-out cells
            else:  # leaky: include the masked middle week(s)
                keep_tps = list(range(lo, hi + 1))
            sel = np.where((cell_type == k) & np.isin(timepoint, keep_tps))[0]
            if sel.size < args.min_cells:
                continue
            Xsub = X.iloc[sel]
            tp_sub = timepoint[sel]
            edges, intercepts = fit_bin(Xsub, tp_sub, mask, gene_to_idx,
                                        alpha=args.alpha, detrend=True,
                                        alpha_mode=args.alpha_mode,
                                        beta_prior_by_target=pick_prior(k, (lo, hi)))
            if not edges:
                continue
            # gamma from endpoint displacement (t_lo -> t_hi), dt = hi-lo; endpoints are
            # train tps in BOTH variants, so gamma itself never sees a held-out marginal.
            gamma = calibrate_gamma(X, timepoint, edges, intercepts, lo, hi, dt=float(hi - lo),
                                    component=args.component)
            artifacts[(k, (lo, hi))] = {
                "edges": edges, "c": intercepts, "gamma": gamma, "n_cells": int(sel.size),
            }
            tag = f" (merged, skips {masked_mid})" if masked_mid else ""
            print(f"  bin type={k:>16} interval={(lo, hi)} cells={sel.size:4d} "
                  f"covered={len(edges):4d} gamma={float(gamma):.3f}{tag}")

    # tag the filename so clean / clean+prior / fixed-vs-adaptive-alpha artifacts coexist
    suffix = args.variant
    if args.prior != "none":
        amode = "aone_pn" if args.alpha_mode == "one_pn" else f"a{args.alpha:g}"
        suffix += f"_prior-{args.prior}_{amode}"
    if args.component != "full":
        suffix += f"_gcal-{args.component}"
    out = args.out or os.path.join(ROOT, "results", "pancreatic_grn.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"genes": genes,
                     "mode": f"interval_foldaware_{suffix}",
                     "prior": args.prior, "alpha_mode": args.alpha_mode, "alpha": args.alpha,
                     "gamma_component": args.component, "component": args.component,
                     "intervals": intervals,
                     "train_timepoints": TRAIN_TPS,
                     "heldout_timepoints": VAL_TPS,
                     "artifacts": artifacts}, f)
    print(f"[save] {len(artifacts)} bins -> {out}")


if __name__ == "__main__":
    main()
