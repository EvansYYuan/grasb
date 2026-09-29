#!/usr/bin/env python
"""Build the fold-aware zebrafish pooled/adaptive regulon artifact (gamma A x)."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_grn_drift import calibrate_gamma, fit_bin, prior_from_selection  # noqa: E402

ZF_CONFIG = ROOT / "config" / "zebrafish"
TRAIN = [3.3, 3.8, 4.3, 4.7, 6.0, 8.0, 10.0, 11.0, 12.0]
HELD_OUT = [5.3, 7.0, 9.0]


def load_mask(path: Path, genes: np.ndarray) -> dict[str, list[str]]:
    """Case-safe CellOracle TFinfo conversion with deterministic parent ordering."""
    tf = pd.read_parquet(path).drop(columns=["peak_id"])
    tf["gene_short_name"] = tf.gene_short_name.astype(str).str.upper()
    rename = {c: str(c).upper() for c in tf.columns if c != "gene_short_name"}
    tf = tf.rename(columns=rename).groupby("gene_short_name", sort=True).sum()
    hvg = set(genes)
    mask = {}
    for target, row in tf.iterrows():
        if target not in hvg:
            continue
        parents = sorted({p for p in row.index[row.to_numpy() > 0] if p in hvg and p != target})
        if parents:
            mask[target] = parents
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expression", type=Path, default=ROOT / "data/zebrafish_expression.csv")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data/zebrafish_metadata.csv")
    ap.add_argument("--groups", type=Path, default=ROOT / "data/zebrafish_groups.csv")
    ap.add_argument("--segments", type=Path, default=ZF_CONFIG / "segment_map.csv")
    ap.add_argument("--base-grn", type=Path, default=ROOT / "data/zebrafish_promoter_base_grn.parquet")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--prior-rare-threshold", type=int, default=200)
    ap.add_argument("--max-bins", type=int, default=0, help="For smoke tests; 0 builds all eligible bins.")
    ap.add_argument("--out", type=Path, default=ROOT / "results/zebrafish_grn.pkl")
    args = ap.parse_args()

    raw = pd.read_csv(args.expression, index_col=0)
    raw.columns = raw.columns.astype(str).str.upper()
    if raw.columns.has_duplicates:
        raise ValueError("Case-normalized expression symbols collide.")
    X = np.log1p(raw.astype(np.float32))
    meta = pd.read_csv(args.metadata, index_col=0).reindex(X.index)
    groups = pd.read_csv(args.groups, index_col=0).reindex(X.index)
    if meta.HPF.isna().any() or groups.segment.isna().all():
        raise ValueError("Metadata alignment failed.")
    time = meta.HPF.astype(float).to_numpy()
    if set(np.unique(time)) & set(HELD_OUT) and set(TRAIN) & set(HELD_OUT):
        raise AssertionError("Train/held-out time sets overlap.")
    segmap = pd.read_csv(args.segments)
    seg_to_lineage = dict(zip(segmap.segment.astype(float), segmap.lineage))
    segment = pd.to_numeric(groups.segment, errors="coerce").to_numpy()
    lineage = np.array([seg_to_lineage.get(s) for s in segment], dtype=object)
    genes = X.columns.to_numpy()
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    mask = load_mask(args.base_grn, genes)
    intervals = list(zip(TRAIN[:-1], TRAIN[1:]))
    train_mask = np.isin(time, TRAIN)

    # Type-temporal priors, with an interval-wide selected-lineage fallback for rare types.
    type_priors = {}
    for k in segmap.lineage:
        sel = np.where((lineage == k) & train_mask)[0]
        if len(sel) >= args.prior_rare_threshold:
            type_priors[k] = prior_from_selection(X, time, sel, mask, gene_to_idx,
                                                  alpha=args.alpha, detrend=True)
    interval_priors = {}
    selected = np.isin(lineage, segmap.lineage)
    for iv in intervals:
        sel = np.where(selected & np.isin(time, iv) & train_mask)[0]
        if len(sel) >= args.min_cells:
            interval_priors[iv] = prior_from_selection(X, time, sel, mask, gene_to_idx,
                                                       alpha=args.alpha, detrend=True)

    artifacts, diagnostics = {}, []
    for k in segmap.lineage:
        for lo, hi in intervals:
            sel = np.where((lineage == k) & np.isin(time, [lo, hi]) & train_mask)[0]
            if len(sel) < args.min_cells:
                continue
            prior = type_priors.get(k, interval_priors.get((lo, hi)))
            edges, intercepts = fit_bin(X.iloc[sel], time[sel], mask, gene_to_idx,
                                        alpha=args.alpha, detrend=True, alpha_mode="one_pn",
                                        beta_prior_by_target=prior)
            if not edges:
                continue
            gamma = calibrate_gamma(X.iloc[sel], time[sel], edges, intercepts,
                                    lo, hi, dt=hi-lo, component="regulon")
            artifacts[(k, (lo, hi))] = {"edges": edges, "c": intercepts,
                                         "gamma": gamma, "n_cells": int(len(sel))}
            fanin = np.array([len(v[0]) for v in edges.values()])
            beta = np.concatenate([v[1] for v in edges.values()])
            diagnostics.append({"lineage": k, "lo": lo, "hi": hi, "n_cells": len(sel),
                                "covered_targets": len(edges), "edges": len(beta),
                                "median_fanin": float(np.median(fanin)),
                                "max_p_over_n": float(fanin.max()/len(sel)),
                                "beta_l2": float(np.linalg.norm(beta)), "gamma": float(gamma),
                                "prior": "type" if k in type_priors else "interval"})
            print(diagnostics[-1])
            if args.max_bins and len(artifacts) >= args.max_bins:
                break
        if args.max_bins and len(artifacts) >= args.max_bins:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"genes": genes, "mode": "zebrafish_interval_foldaware_type_ivl_aone_pn_gcal-regulon",
               "intervals": intervals,
               "train_timepoints": TRAIN, "heldout_timepoints": HELD_OUT,
               "alpha": args.alpha, "alpha_mode": "one_pn", "prior": "type_ivl",
               "gamma_component": "regulon", "component": "regulon",
               "artifacts": artifacts}
    with args.out.open("wb") as fh:
        pickle.dump(payload, fh)
    pd.DataFrame(diagnostics).to_csv(args.out.with_suffix(".diagnostics.csv"), index=False)
    args.out.with_suffix(".manifest.json").write_text(json.dumps({
        "artifact": str(args.out), "bins": len(artifacts), "genes": len(genes),
        "candidate_targets": len(mask), "candidate_edges": sum(map(len, mask.values())),
        "train_timepoints": TRAIN, "heldout_timepoints": HELD_OUT,
        "field": "gamma A x", "attractor_term": False}, indent=2) + "\n")
    print(f"saved {len(artifacts)} bins to {args.out}")


if __name__ == "__main__":
    main()
