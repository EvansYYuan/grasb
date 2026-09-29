#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reusable fitting routines for the gene-space regulatory velocity in GRASB.

For the manuscript configuration, use ``build_fold_aware_grn.py``. This module
also retains a simple all-timepoint command-line builder for method development.
It turns the pancreatic HVG data + CellOracle's human promoter base GRN into the
sparse per-(cell type, interval) artifacts that ARTEMIS's training loss consumes
online at loss.py:55:

    b_GRN(x, t) = gamma_t * [ (A_{k,t} - I) x + c_{k,t} ]
    L_GRN       = lambda * E|| P( b_theta - b_GRN ) ||^2

Nothing here imports `celloracle` as a package. Instead we (a) read its
shipped base-GRN parquet as a data asset and (b) reimplement two tiny recipes from
its source, cited inline. The only extra dependency over the bcdsb env is `pyarrow`
(to read the parquet).

Run modes (incremental, per the agreed v0 plan):
  --mode whole     : one GRN per cell type over ALL its cells, no detrend, keep
                     CellOracle's intercept-dropping convention. This reproduces
                     CellOracle's coef_matrix_per_cluster and is the plumbing /
                     base-GRN-match sanity check.
  --mode interval  : the real prior -- per (type, interval = t_n U t_{n+1}),
                     detrend per timepoint to estimate slopes A, recover intercept
                     c on the original scale (attractor lives in real expr space),
                     and calibrate gamma_t. This is what feeds ARTEMIS.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Module 1 -- load_expression
#   WHAT: load the log1p HVG matrix the SB lives in (cells x genes, gene symbols).
#   FROM: our own reproduction output
#         (`data/pancreatic_preprocessed.csv`). Last col is "time".
#   WHY : b_GRN is fit in the SAME raw log1p HVG space b_theta is penalized against
#         (decided: raw HVG, no KNN imputation -- plan discussion / §2 remove-VAE).
# ---------------------------------------------------------------------------
def load_expression(preprocessed_csv):
    df = pd.read_csv(preprocessed_csv, index_col=0)
    time = df["time"].astype(int).values            # CellWeek per cell
    X = df.drop(columns=["time"])                   # cells x genes (symbols)
    genes = X.columns.to_numpy()
    return X.astype(np.float32), genes, time, X.index.to_numpy()


# ---------------------------------------------------------------------------
# Module 2 -- attach_labels
#   WHAT: align per-cell (cell_type, timepoint) to the expression rows.
#   FROM: data/pancreatic/GSE114412_Stage_5.all.cell_metadata.tsv, joined exactly as
#         reproduction/preprocessing/preprocess_pancreatic.py:74-92 (index =
#         "library.barcode"; columns Assigned_cluster, CellWeek).
#   WHY : pi(k|x) is the observed annotation, one-hot, never inferred.
# ---------------------------------------------------------------------------
def attach_labels(barcodes, meta_tsv):
    meta = pd.read_csv(meta_tsv, sep="\t", index_col="library.barcode")
    meta = meta.reindex(barcodes)                   # align to expression row order
    if meta["Assigned_cluster"].isna().any():
        raise ValueError("Some expression barcodes have no Assigned_cluster.")
    cell_type = meta["Assigned_cluster"].to_numpy()
    timepoint = meta["CellWeek"].astype(int).to_numpy()
    return cell_type, timepoint


# ---------------------------------------------------------------------------
# Module 3 -- load_base_grn_mask  (the topology mask M)
#   WHAT: build {target_gene -> [candidate parent TFs]}, then keep only edges whose
#         BOTH endpoints are in HVG (the HVG-internal sub-network).
#   FROM: - data asset: CellOracle/celloracle/data/promoter_base_GRN/
#                       hg38_TFinfo_dataframe_gimmemotifsv5_fpr2_*.parquet
#                       (loaded by celloracle.data.load_human_promoter_base_GRN;
#                        we read the parquet directly).
#         - the TFinfo -> TFdict recipe reimplemented from
#           celloracle/network/net_core.py:163-168 (drop peak_id, groupby
#           gene_short_name, sum, keep nonzero columns per target).
#   WHY : M is a swappable input; the GRN is computed on whole genes then
#         filtered to the HVG-internal sub-network (b_GRN is read from HVG state x).
#         scATAC not needed: M only gates candidate edges; signs come from regression.
# ---------------------------------------------------------------------------
def load_base_grn_mask(parquet_path, hvg_genes):
    tfinfo = pd.read_parquet(parquet_path)          # cols: peak_id, gene_short_name, <TF columns 0/1>
    # --- net_core.py:163-168 ---
    tmp = tfinfo.drop(columns=["peak_id"])
    tmp = tmp.groupby(by="gene_short_name").sum()
    tfdict = dict(tmp.apply(lambda r: r[r > 0].index.to_numpy(), axis=1))
    # Restrict to the HVG-internal sub-network.
    hvg = set(hvg_genes.tolist())
    mask = {}
    for target, parents in tfdict.items():
        if target not in hvg:
            continue
        keep = [p for p in parents if p in hvg and p != target]   # self-edge removed (net_core / oracle_GRN _correct_coef_table)
        if keep:
            mask[target] = keep
    return mask


# ---------------------------------------------------------------------------
# Module 4 -- fit_target  (signed coefficients A = M (.) beta_hat, plus intercept c)
#   WHAT: for one target gene over one cell subset, regress its level on its
#         mask-licensed parent levels -> signed betas + intercept.
#   FROM: CellOracle's simulation-coef recipe -- plain Ridge(alpha=1) per target,
#         oracle_GRN.py:55-115 (_getCoefMatrix) and regression_models.py:82-146
#         (get_bagging_ridge_coefs). We use the single-Ridge simulation path, NOT
#         the bagging/p-value Net path (that path is for network viz, not dynamics).
#   OUR DEVIATIONS (deliberate, documented in README):
#     1. KEEP the intercept. _getCoefMatrix stores only model.coef_ (line 93) and
#        the paper (p.13) drops C on purpose because it works in pure shift space.
#        We need c for the attractor x* = (I-A)^-1 c, so we retain it.
#     2. DETREND. CellOracle does not. Pancreatic cross-week trend
#        must be removed before pooling endpoints or it leaks into A as fake
#        coregulation. We center per timepoint to get SLOPES, then recover c on the
#        ORIGINAL scale so the attractor stays in real expression coordinates.
# ---------------------------------------------------------------------------
def fit_target(Xsub, target, parents, timepoint_sub, alpha=1.0, detrend=False,
               beta_prior=None):
    """Ridge for one target on its mask-licensed parents.

    `beta_prior` (Block 2.1 Part B, 2026-07-02): the borrow-strength / partial-pooling
    fix. When given (a coefficient vector aligned to `parents`), the objective becomes
    ridge-toward-a-prior `min ||y - Xb||^2 + alpha||b - beta_prior||^2` instead of the
    default shrink-to-ZERO `min ||y - Xb||^2 + alpha||b||^2`. Mechanically: fit ridge on
    the RESIDUAL target `y - X.beta_prior`, then add beta_prior back. Same adaptive alpha
    schedule as the baseline, so the ONLY thing that changes is the shrinkage *center* --
    thin bins fall back to an informative pooled prior (empirical Bayes / James-Stein)
    rather than to 0, which cuts variance without homogenising bins toward a dead field.
    beta_prior=None recovers the exact prior CellOracle-style behavior.
    """
    parents = list(parents)
    Xp = Xsub[parents].to_numpy(dtype=np.float64)   # cells x n_parents
    y = Xsub[target].to_numpy(dtype=np.float64)     # cells
    b0 = None if beta_prior is None else np.asarray(beta_prior, dtype=np.float64)
    if b0 is not None and b0.shape[0] != Xp.shape[1]:
        raise ValueError(f"beta_prior len {b0.shape[0]} != n_parents {Xp.shape[1]} for {target}")

    if detrend:
        # Center per timepoint so slopes are free of developmental trend.
        Xp_c = Xp.copy()
        y_c = y.copy()
        for t in np.unique(timepoint_sub):
            m = timepoint_sub == t
            Xp_c[m] -= Xp[m].mean(axis=0, keepdims=True)
            y_c[m] -= y[m].mean()
        model = Ridge(alpha=alpha, fit_intercept=False, random_state=123)
        if b0 is None:
            model.fit(Xp_c, y_c)
            beta = model.coef_
        else:
            model.fit(Xp_c, y_c - Xp_c @ b0)      # shrink residual toward 0 == shrink beta toward b0
            beta = b0 + model.coef_
        # recover intercept on ORIGINAL scale so attractor lives in real expr space:
        # c = ybar - beta . xbar   (bin-level means, uncentered)
        c = y.mean() - beta @ Xp.mean(axis=0)
    else:
        # CellOracle-faithful: Ridge with intercept, slopes on raw levels
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=123)
        if b0 is None:
            model.fit(Xp, y)
            beta = model.coef_
            c = model.intercept_
        else:
            model.fit(Xp, y - Xp @ b0)
            beta = b0 + model.coef_
            c = model.intercept_

    return np.asarray(parents), beta.astype(np.float32), np.float32(c)


# ---------------------------------------------------------------------------
# Module 5 -- fit_bin  (drive Module 4 over all covered targets in one bin)
#   WHAT: produce the sparse edge list for one (type, interval) cell subset.
#   FROM: loop structure mirrors _getCoefMatrix's per-gene loop (oracle_GRN.py:97-108).
#   OUTPUT: {target_idx -> (parent_idxs, betas)}, c[target_idx]
#         with indices in HVG column order (0..n_genes-1) so the jax side gathers by index.
# ---------------------------------------------------------------------------
def _adaptive_alpha(alpha_mode, base, p, n):
    """Per-target Ridge strength (Step 3 thin-bin handling; SUMMARY §4.4).

    Thin bins have rho(A) >= 1 because targets are regressed on p parents with
    p >= n (overfitting), inflating the slopes. We do NOT prune (that deletes
    validated biology, §4.3); instead we scale the Ridge penalty up with the
    per-target predictor/sample ratio p/n so underdetermined fits shrink:

      fixed   : a = base                  (CellOracle default; the lambda-baseline)
      one_pn  : a = base * (1 + p/n)       (smooth, always-on; the chosen rule)

    `one_pn` is self-deactivating on well-populated bins (p/n -> 0 => a -> base,
    coefficients and rho unchanged) and pulls every thin bin's rho below 1 with
    margin. The threshold form max(1, p/n) was tried and REJECTED: it only touches
    the minority of targets with p >= n, leaving rho ~unchanged (experiment_adaptive_ridge.py).
    """
    if alpha_mode == "fixed":
        return base
    if alpha_mode == "one_pn":
        return base * (1.0 + p / max(n, 1))
    raise ValueError(f"unknown alpha_mode {alpha_mode!r}")


def fit_bin(Xsub, timepoint_sub, mask, gene_to_idx, alpha=1.0, detrend=False,
            alpha_mode="fixed", beta_prior_by_target=None):
    """Fit every covered target in one (type, interval) cell subset.

    `beta_prior_by_target` (Block 2.1 Part B): optional {target_idx -> beta0 vector}
    borrow-strength prior (see fit_target). None -> baseline shrink-to-zero. The prior's
    beta0 for a target is aligned to that target's mask-filtered parent order, which is
    deterministic across bins because the topology mask M is global, so a pooled
    fit's coefficients drop straight into any bin for the same target.
    """
    n = Xsub.shape[0]
    edges = {}          # target_idx -> (parent_idxs[int32], betas[float32])
    intercepts = {}     # target_idx -> c
    for target, parents in mask.items():
        # only parents actually present as columns (defensive; mask already HVG-filtered)
        parents = [p for p in parents if p in gene_to_idx]
        if not parents:
            continue
        a = _adaptive_alpha(alpha_mode, alpha, len(parents), n)
        g = gene_to_idx[target]
        b0 = None if beta_prior_by_target is None else beta_prior_by_target.get(g)
        p_names, beta, c = fit_target(Xsub, target, parents, timepoint_sub,
                                      alpha=a, detrend=detrend, beta_prior=b0)
        edges[g] = (np.array([gene_to_idx[p] for p in p_names], dtype=np.int32), beta)
        intercepts[g] = c
    return edges, intercepts


def prior_from_selection(X, timepoint, sel, mask, gene_to_idx, alpha=1.0,
                         detrend=True):
    """Fit a well-determined pooled prior beta0 on a large cell selection.

    Returns {target_idx -> beta0 vector} in the same parent order fit_bin uses, ready to
    pass as `beta_prior_by_target`. Uses FIXED base alpha (no adaptive inflation): the
    pool is large, so the prior itself should be data-driven, not shrunk. This is the
    borrow-strength source for Block 2.1 Part B; the pool = which cells `sel` picks
    (global = all cells; interval-sibling = all types at one interval; type-temporal =
    one type across intervals).
    """
    edges, _ = fit_bin(X.iloc[sel], timepoint[sel], mask, gene_to_idx,
                       alpha=alpha, detrend=detrend, alpha_mode="fixed")
    return {g: np.asarray(beta, dtype=np.float64) for g, (pidx, beta) in edges.items()}


# ---------------------------------------------------------------------------
# Module 6 -- calibrate_gamma  (the units knob gamma_t, so lambda stays pure trust)
#   WHAT: gamma = V_obs / V_raw on covered genes.
#         V_obs = || mean expr at t_{n+1} - mean expr at t_n || / dt   (population
#                 mean displacement of that type; no lineage tracing available).
#         V_raw = mean over bin cells of || (A-I)x + c || on covered genes.
#   WHY : (A-I)x+c is a weak raw field (paper p.16: coefs in [-1,1]); gamma rescales
#         relaxation SPEED to match observed displacement-per-dt without moving the
#         attractor, keeping lambda a pure trust dial.
#   NOTE: only meaningful in --mode interval. One scalar per (type, interval);
#         a single shared gamma is the fallback.
# ---------------------------------------------------------------------------
def calibrate_gamma(Xsub, timepoint_sub, edges, intercepts, t_lo, t_hi, dt=1.0,
                    component="full"):
    """gamma = V_obs / V_raw, where V_raw MUST be measured on the SAME field that will be
    DEPLOYED at runtime (engine/grn_drift.py `b_grn`), else the units knob is calibrated
    against a quantity we then discard.

    component="full"    -> V_raw = mean||(A-I)x + c||   (matches bg = g*(Ax + c - x))
    component="regulon" -> V_raw = mean||A x||          (matches bg = g*Ax, "strip -x+c")

    BUG THIS FIXES (2026-07-15): we deploy --grn-target regulon (g*Ax) but gamma was
    calibrated on the FULL field. Measured: the regulon field is ~3x smaller than the
    full field, so the deployed b_GRN came out ~5x SMALLER than the observed weekly
    displacement (median |b_GRN|/V_obs = 0.19x over 51 bins). The L2 drift penalty then
    faithfully drags the bridge's drift to that too-small target => the bridge undershoots
    the week-long jump => the +2.43 W2 cost of any lambda>0, and the lambda switch
    (lam=0.25 already pins the drift at b_GRN's scale; more lambda cannot pin harder).
    Fixing the scale does NOT change b_GRN's DIRECTION, so the biology should survive.
    """
    covered = np.array(sorted(edges.keys()), dtype=np.int64)
    if covered.size == 0:
        return np.float32(1.0)
    Xv = Xsub.to_numpy(dtype=np.float64)
    # V_obs: population mean displacement on covered genes
    lo = Xv[timepoint_sub == t_lo].mean(axis=0)
    hi = Xv[timepoint_sub == t_hi].mean(axis=0)
    V_obs = np.linalg.norm((hi - lo)[covered]) / dt
    # V_raw: mean || deployed field || over bin cells, covered genes only
    acc = np.zeros((Xv.shape[0], covered.size))
    for j, g in enumerate(covered):
        p_idx, beta = edges[g]
        # p_idx / g are HVG-global indices == column positions here (Xsub is full HVG)
        Ax = Xv[:, p_idx] @ beta
        acc[:, j] = Ax if component == "regulon" else (Ax + intercepts[g] - Xv[:, g])
    V_raw = np.mean(np.linalg.norm(acc, axis=1))
    gamma = V_obs / V_raw if V_raw > 0 else 1.0
    return np.float32(gamma)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preprocessed",
                    default="data/pancreatic_preprocessed.csv")
    ap.add_argument("--meta",
                    default="data/pancreatic/GSE114412_Stage_5.all.cell_metadata.tsv")
    ap.add_argument("--base-grn",
                    default="CellOracle/celloracle/data/promoter_base_GRN/"
                            "hg38_TFinfo_dataframe_gimmemotifsv5_fpr2_threshold_10_20210630.parquet")
    ap.add_argument("--mode", choices=["whole", "interval"], default="interval")
    ap.add_argument("--alpha", type=float, default=1.0,    # CellOracle default (oracle_core.fit_GRN_for_simulation)
                    help="Base Ridge regularization strength.")
    ap.add_argument("--alpha-mode", choices=["fixed", "one_pn"], default="one_pn",
                    help="Thin-bin handling (SUMMARY §4.4). 'fixed'=CellOracle default a=alpha; "
                         "'one_pn'=adaptive a=alpha*(1+p/n) so underdetermined bins shrink "
                         "and rho(A)<1. Default one_pn for the prior; use fixed for whole-mode "
                         "CellOracle-faithfulness checks.")
    ap.add_argument("--min-cells", type=int, default=30,
                    help="Skip (type[,interval]) bins thinner than this.")
    ap.add_argument("--out", default="results/pancreatic_grn_unmasked.pkl")
    args = ap.parse_args()

    X, genes, time, barcodes = load_expression(args.preprocessed)
    cell_type, timepoint = attach_labels(barcodes, args.meta)
    mask = load_base_grn_mask(args.base_grn, genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    n_cov = len(set(mask.keys()) & set(genes.tolist()))
    print(f"[load] cells={X.shape[0]} genes={X.shape[1]} "
          f"types={len(np.unique(cell_type))} timepoints={sorted(np.unique(timepoint))}")
    print(f"[mask] HVG-internal covered targets={n_cov} "
          f"(edges total={sum(len(v) for v in mask.values())})")

    artifacts = {}   # (type, interval) -> dict(edges, c, gamma, n_cells)
    detrend = (args.mode == "interval")
    # whole mode is the CellOracle-faithful sanity path -> keep its base alpha exactly.
    alpha_mode = "fixed" if args.mode == "whole" else args.alpha_mode
    print(f"[fit] mode={args.mode} detrend={detrend} alpha={args.alpha} alpha_mode={alpha_mode}")

    if args.mode == "whole":
        bins = [(k, None, np.where(cell_type == k)[0]) for k in np.unique(cell_type)]
    else:
        ts = sorted(np.unique(timepoint))
        intervals = list(zip(ts[:-1], ts[1:]))   # (t_n, t_{n+1})
        bins = []
        for k in np.unique(cell_type):
            for (lo, hi) in intervals:
                sel = np.where((cell_type == k) & np.isin(timepoint, [lo, hi]))[0]
                bins.append((k, (lo, hi), sel))

    for k, interval, sel in bins:
        if sel.size < args.min_cells:
            continue
        Xsub = X.iloc[sel]
        tp_sub = timepoint[sel]
        edges, intercepts = fit_bin(Xsub, tp_sub, mask, gene_to_idx,
                                    alpha=args.alpha, detrend=detrend,
                                    alpha_mode=alpha_mode)
        if not edges:
            continue
        if interval is None:
            gamma = np.float32(1.0)
        else:
            gamma = calibrate_gamma(X, timepoint, edges, intercepts,
                                    interval[0], interval[1])
        artifacts[(k, interval)] = {
            "edges": edges, "c": intercepts, "gamma": gamma, "n_cells": int(sel.size),
        }
        print(f"  bin type={k:>16} interval={interval} cells={sel.size:4d} "
              f"covered={len(edges):4d} gamma={float(gamma):.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"genes": genes, "mode": args.mode, "artifacts": artifacts}, f)
    print(f"[save] {len(artifacts)} bins -> {args.out}")


if __name__ == "__main__":
    main()
