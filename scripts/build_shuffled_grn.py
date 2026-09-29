#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a SHUFFLED-topology control GRN for the regulon-penalty honesty check.

Takes the real b_GRN artifact (`pancreatic_b_grn_fix.pkl`) and replaces each
target's PARENT identities with a random, degree-preserving set drawn from the
same parent pool — while keeping that target's real per-bin β values, its γ, and
c untouched. One shared random reassignment per target (applied to every bin), so
the control mirrors the real structure (shared topology M, per-bin weights) and
differs ONLY in the biological identity of the edges.

Rationale: the per-type drivers (SST→δ, SLC30A8→β, NEUROG3→commitment) under the
regulon penalty are partly "by construction" (the penalty injects each type's
regulon). Training the SAME regulon penalty against this random-wiring GRN tests
whether that lineage-correctness is the REAL GRN's wiring (→ specificity should
collapse) or would arise from any TF-structured field (→ it would survive).

Note: the regulon penalty (`--grn-target regulon`) uses only γAx, so c is
irrelevant here; we copy it through unchanged for a drop-in artifact.
"""
import argparse, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-pkl", default="results/pancreatic_grn.pkl")
    ap.add_argument("--out", default="results/pancreatic_grn_shuffled.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = pickle.load(open(args.in_pkl, "rb"))
    artifacts = d["artifacts"]
    rng = np.random.default_rng(args.seed)

    # parent pool = union of all parent indices over all bins (keep shuffled parents "TF-like")
    pool = set()
    fanin = {}        # target idx -> fan-in (shared across bins; assert consistency)
    for v in artifacts.values():
        for g, (pidx, _) in v["edges"].items():
            pool.update(int(p) for p in pidx)
            if g in fanin and fanin[g] != len(pidx):
                # topology not perfectly shared for this target; use the max fan-in seen
                fanin[g] = max(fanin[g], len(pidx))
            else:
                fanin.setdefault(g, len(pidx))
    pool = np.array(sorted(pool))
    print(f"[load] bins={len(artifacts)} targets={len(fanin)} parent-pool={pool.size}")

    # one shared random parent set per target (degree-preserving, excl self)
    rand_parents = {}
    for g, k in fanin.items():
        cand = pool[pool != g]
        rand_parents[g] = rng.choice(cand, size=min(k, cand.size), replace=False).astype(np.int32)

    # rewrite every bin: keep β (truncate/pad to the random set length), swap parents
    n_edges = 0
    for v in artifacts.values():
        new_edges = {}
        for g, (pidx, beta) in v["edges"].items():
            rp = rand_parents[g][: len(beta)]            # align to this bin's β length
            new_edges[g] = (rp.copy(), np.asarray(beta, dtype=np.float32).copy())
            n_edges += len(rp)
        v["edges"] = new_edges

    d["mode"] = d.get("mode", "interval") + "_shuffled"
    pickle.dump(d, open(args.out, "wb"))
    print(f"[save] shuffled-topology GRN ({n_edges} edge-slots rewritten) -> {args.out}")


if __name__ == "__main__":
    main()
