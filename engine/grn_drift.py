# -*- coding: utf-8 -*-
"""Online GRN drift `b_GRN` for the λ-leg (plan §5–§8).

Loads the offline artifact produced by `build_grn_drift.py`
(`{genes, mode, artifacts}`, artifacts keyed by `(celltype_str, (t_lo,t_hi))`,
each `{edges: {target_idx -> (parent_idx[], beta[])}, c: {target_idx->float},
gamma: float, n_cells}`) and turns it into JAX-friendly arrays so the training
loss can evaluate, per particle x and integration step k,

    b_GRN(x, t) = gamma_bin * ( (A_bin - I) x + c_bin )                       (plan §5)
    A_bin x [g] = sum_{p in parents(g)} beta_bin[g,p] * x[p]                  (sparse)

projected onto covered genes (those with >=1 declared regulator). The penalty
`lambda * mean_{covered g} (b_theta[g] - b_GRN[g])^2` is added in loss.py; with
lambda=0 nothing here runs and the engine is the §5 tuned baseline exactly.

Representation. Topology M is shared across bins (only signs/weights vary), so we
build ONE global edge list (`tgt_idx`, `par_idx`, length E ~ 7e4) and per-bin
weight rows `beta[B,E]`, plus `c[B,G]`, `gamma[B]`. A particle's bin is
(cell-type, interval): cell type from its observed `Assigned_cluster` label
(plan §7, one-hot, never inferred), interval from the integration step via
`step_to_interval`. Skipped (type,interval) bins (the 9 biologically-absent ones)
map to bin id -1 → the loss applies no penalty to those particles (P projects out).
"""

import numpy as np


class GRNDrift:
    """Host-side container + JAX evaluator for `b_GRN`.

    Build with `GRNDrift.from_pkl(path)`. The arrays are plain numpy on the host;
    `to_jax()` converts the field arrays to `jnp` once (call after jax import).
    `b_grn(pos, bin_id)` is a pure function of jnp arrays usable inside jit.
    """

    def __init__(self, genes, type_to_id, interval_lo, bin_lookup,
                 tgt_idx, par_idx, beta, c, gamma, covered_mask, interval_hi=None):
        self.genes = genes
        self.n_genes = len(genes)
        self.type_to_id = type_to_id
        self.id_to_type = {v: k for k, v in type_to_id.items()}
        self.n_types = len(type_to_id)
        self.interval_lo = list(interval_lo)          # e.g. [0,1,2,3,4,5,6]
        # upper bounds; default = lo+1 (unit consecutive intervals, backward compatible).
        # A fold-aware build that masks a held-out week has a MERGED interval, e.g.
        # lo=[0,1,2,4,5], hi=[1,2,4,5,7] where [2,4) and [5,7) span the masked t3/t6.
        self.interval_hi = ([lo + 1 for lo in interval_lo] if interval_hi is None
                            else list(interval_hi))
        self.n_intervals = len(self.interval_lo)
        self.bin_lookup = bin_lookup                  # (n_types, n_intervals) int, -1 if skipped
        self.n_bins = beta.shape[0]
        # field arrays (numpy until to_jax()):
        self.tgt_idx = tgt_idx                        # (E,) int32  global gene idx of each edge's target
        self.par_idx = par_idx                        # (E,) int32  global gene idx of each edge's parent
        self.beta = beta                              # (B, E) float32
        self.c = c                                    # (B, G) float32
        self.gamma = gamma                            # (B,) float32
        self.covered_mask = covered_mask              # (G,) float32  1 for covered targets
        # penalty target: "full" → γ((A−I)x+c) (attractor field); "regulon" → γAx only
        # (strip the −x+c degradation/intercept term to test whether the SIGNED regulon
        # imprints, SUMMARY §5.4 A-vs-B). Static attr → branch resolves at jit trace time.
        self.component = "full"
        self._is_jax = False

    # ------------------------------------------------------------------ build
    @staticmethod
    def from_pkl(path):
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        genes = np.asarray(d["genes"])
        G = len(genes)
        artifacts = d["artifacts"]

        keys = list(artifacts.keys())                  # [(type_str, (lo,hi)), ...]
        type_strs = sorted({k[0] for k in keys})
        type_to_id = {t: i for i, t in enumerate(type_strs)}
        # interval id = rank by lower bound. Intervals must TILE [min_lo, max_hi)
        # contiguously & disjointly (checked below), but no longer need to be unit
        # width: a fold-aware build merges across a masked week (e.g. [2,4), [5,7)).
        # Preserve real-valued biological time. Pancreatic stages happen to be integers,
        # while zebrafish HPF includes 3.3, 3.8, 4.3, ...; coercing to int collapses
        # distinct intervals and corrupts lookup.
        declared = d.get("intervals")
        lohis = (sorted((float(lo), float(hi)) for lo, hi in declared) if declared is not None
                 else sorted({(float(k[1][0]), float(k[1][1])) for k in keys}))
        interval_lo = [lo for lo, _ in lohis]
        interval_hi = [hi for _, hi in lohis]
        for i in range(len(lohis) - 1):
            assert interval_hi[i] == interval_lo[i + 1], \
                f"intervals do not tile contiguously: {lohis}"
        ivl_to_id = {lo: i for i, lo in enumerate(interval_lo)}
        n_types, n_intervals = len(type_to_id), len(interval_lo)

        # --- global edge list = union over bins of (target_idx, parent_idx) ---
        edge_pos = {}                                  # (g,p) -> e
        covered = set()
        for v in artifacts.values():
            for g, (pidx, beta_g) in v["edges"].items():
                covered.add(int(g))
                for p in pidx:
                    key = (int(g), int(p))
                    if key not in edge_pos:
                        edge_pos[key] = len(edge_pos)
        E = len(edge_pos)
        # deterministic order by (target, parent)
        ordered = sorted(edge_pos.keys())
        edge_pos = {k: i for i, k in enumerate(ordered)}
        tgt_idx = np.array([g for (g, _) in ordered], dtype=np.int32)
        par_idx = np.array([p for (_, p) in ordered], dtype=np.int32)

        B = len(keys)
        beta = np.zeros((B, E), dtype=np.float32)
        c = np.zeros((B, G), dtype=np.float32)
        gamma = np.zeros((B,), dtype=np.float32)
        bin_lookup = -np.ones((n_types, n_intervals), dtype=np.int32)

        for b, k in enumerate(keys):
            v = artifacts[k]
            gamma[b] = np.float32(v["gamma"])
            for g, cval in v["c"].items():
                c[b, int(g)] = np.float32(cval)
            for g, (pidx, beta_g) in v["edges"].items():
                gi = int(g)
                for p, bt in zip(pidx, beta_g):
                    beta[b, edge_pos[(gi, int(p))]] = np.float32(bt)
            bin_lookup[type_to_id[k[0]], ivl_to_id[float(k[1][0])]] = b

        covered_mask = np.zeros((G,), dtype=np.float32)
        covered_mask[sorted(covered)] = 1.0

        obj = GRNDrift(genes, type_to_id, interval_lo, bin_lookup,
                       tgt_idx, par_idx, beta, c, gamma, covered_mask,
                       interval_hi=interval_hi)
        component = d.get("component", "full")
        if component not in {"full", "regulon"}:
            raise ValueError(f"Unknown GRN component in artifact: {component!r}")
        obj.component = component
        return obj

    # ------------------------------------------------------------- host utils
    def labels_to_type_ids(self, cell_type_strs):
        """Map an array of `Assigned_cluster` strings → type ids (int32).

        Unknown types (not in the GRN) map to -1; the loop below treats any
        particle whose (type,interval) bin is missing as unconstrained.
        """
        return np.array([self.type_to_id.get(str(t), -1) for t in cell_type_strs],
                        dtype=np.int32)

    def step_to_interval(self, steps_num, tp_min=None, tp_max=None):
        """Map each integration step k=0..steps_num → GRN interval id.

        The bridge integrates developmental time linearly from the first to the
        last *observed* marginal (cells_time in datasets.py), which for pancreatic
        spans CellWeek tp_min=0 .. tp_max=7 (endpoints are never held out). The
        developmental tp at step k is τ(k)=tp_min+(k/steps_num)*(tp_max-tp_min).
        Interval id = the tiling cell [lo_i, hi_i) that contains τ, found by
        boundary search on interval_lo. This handles MERGED intervals from a
        fold-aware build (e.g. τ=3 falls in the merged [2,4) that skips a masked
        t3) and reduces to floor(τ)-lo0 for unit consecutive intervals.
        """
        lo0 = self.interval_lo[0]
        tp_min = lo0 if tp_min is None else tp_min
        tp_max = (self.interval_hi[-1]) if tp_max is None else tp_max
        ks = np.arange(steps_num + 1)
        tau = tp_min + (ks / steps_num) * (tp_max - tp_min)
        # id = index of the last lo <= τ  (searchsorted 'right' - 1), clamped in range.
        iv = np.clip(np.searchsorted(self.interval_lo, tau, side="right") - 1,
                     0, self.n_intervals - 1)
        return iv.astype(np.int32)

    def interval_id_for_tp(self, tp):
        """Interval id whose tiling cell [lo,hi) contains integer stage `tp`.
        Boundary search on interval_lo -> correct for MERGED and fractional intervals
        (e.g. zebrafish 4.7 -> 6.0). Reduces to tp-lo0 for unit consecutive intervals."""
        return int(np.clip(np.searchsorted(self.interval_lo, tp, side="right") - 1,
                           0, self.n_intervals - 1))

    def bin_id_array(self):
        """(n_types, n_intervals) int32 lookup; index with [type_id, interval_id]."""
        return self.bin_lookup

    @property
    def sentinel_bin(self):
        """A safe gather index for invalid particles (points at an appended zero row)."""
        return self.n_bins

    # ------------------------------------------------------------------- jax
    def to_jax(self):
        """Convert field arrays to jnp, appending a zero 'sentinel' bin row at
        index n_bins so invalid bin ids (-1 → sentinel) gather zeros."""
        import jax.numpy as jnp
        beta = jnp.concatenate([jnp.asarray(self.beta),
                                jnp.zeros((1, self.beta.shape[1]), self.beta.dtype)], 0)
        c = jnp.concatenate([jnp.asarray(self.c),
                             jnp.zeros((1, self.c.shape[1]), self.c.dtype)], 0)
        gamma = jnp.concatenate([jnp.asarray(self.gamma), jnp.zeros((1,), self.gamma.dtype)], 0)
        self._jax = dict(
            tgt_idx=jnp.asarray(self.tgt_idx), par_idx=jnp.asarray(self.par_idx),
            beta=beta, c=c, gamma=gamma,
            covered_mask=jnp.asarray(self.covered_mask),
        )
        self._is_jax = True
        return self

    def b_grn(self, pos, bin_id):
        """b_GRN for a batch. pos:(batch,G), bin_id:(batch,) with -1 for unconstrained.

        Returns (batch,G); rows of unconstrained particles and non-covered genes
        are zero. Call `to_jax()` first. Pure → safe inside jit.
        """
        import jax.numpy as jnp
        a = self._jax
        b = jnp.where(bin_id < 0, self.sentinel_bin, bin_id)      # clamp invalid → zero row
        beta_b = a["beta"][b]                                      # (batch, E)
        pos_par = pos[:, a["par_idx"]]                            # (batch, E)
        contrib = beta_b * pos_par                                # (batch, E)
        # Ax[:, g] = sum of contrib over edges whose target is g (duplicate idx summed)
        Ax = jnp.zeros_like(pos).at[:, a["tgt_idx"]].add(contrib)  # (batch, G)
        cb = a["c"][b]                                             # (batch, G)
        gb = a["gamma"][b][:, None]                                # (batch, 1)
        if self.component == "regulon":
            bg = gb * Ax                                           # γ A x only (strip −x+c)
        else:
            bg = gb * (Ax + cb - pos)                              # γ((A-I)x + c), scaled
        return bg * a["covered_mask"][None, :]
