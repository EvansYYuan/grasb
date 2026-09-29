#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Decode the LATENT SB drift to gene space and emit the per-type drift CSV that
the biology battery (q_biology_gsea.py, q1q3_rss_sweep.py) consumes.

The latent model's learned drift is  b'(t,z') = f + g.Z(z')  in STANDARDIZED-latent
units. The interpretable gene-space drift is its pushforward through the FROZEN
decoder:
        x = dec(z'*z_std + z_mean)          (gene-space image of the latent pos)
        dx/dt = J_dec(z) . (z_std (.) b')   (chain rule; z = z'*z_std+z_mean)
i.e. a decoder jvp -- the exact analogue of ARTEMIS's "push the latent drift
through the decoder to get per-gene drift", and of the gene-space engine's direct
b_theta. Per (cell type, stage) we average dx/dt over that group's cells and write
(tag, cell_type, stage, gene, drift) rows -- identical schema to
interpret_drift_grn.py --per-type-full-out, so the downstream scripts are reused
verbatim.

Run:  JAX_PLATFORMS=cpu python interpret_latent.py --model output/pancreatic_latent_..._model_params.pkl \
        --tag latent_grn --per-type-full-out output/per_type_drift_full_latent.csv
"""
import argparse
import os
import sys
import csv
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE_SRC = os.path.join(ROOT, "engine")
DEFAULT_DATA = os.path.join(ROOT, "data", "pancreatic_preprocessed.csv")
DEFAULT_META = os.path.join(ROOT, "data", "pancreatic_metadata.tsv")
DEFAULT_GRN = os.path.join(ROOT, "results", "pancreatic_grn.pkl")
TRAIN_TPS = [0, 1, 2, 4, 5, 7]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="latent model_params pkl (from train_latent_grn.py --save).")
    p.add_argument("--vae", default=os.path.join(ROOT, "results", "vae_frozen.pkl"))
    p.add_argument("--joint-vae", default=None,
                   help="If set, use the frozen JOINT (t-conditioned) VAE from this engine "
                        "model_params pkl, read at --t-ref, instead of the plain frozen VAE. "
                        "Must match the --joint-vae/--t-ref used in train_latent_grn.py.")
    p.add_argument("--t-ref", type=float, default=3.5)
    p.add_argument("--grn-pkl", default=DEFAULT_GRN, help="only for type_to_id / labels.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--meta", default=DEFAULT_META)
    p.add_argument("--label-mode", choices=["pancreatic", "segment"], default="pancreatic")
    p.add_argument("--segment-map", default=None)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--f-val", type=float, default=0.0)
    p.add_argument("--g-max", type=float, default=0.5)
    p.add_argument("--g-type", default="triangular")
    p.add_argument("--n-cells", type=int, default=500)
    p.add_argument("--stages", default="0,1,2,3,4,5,6,7")
    p.add_argument("--train-tps", default=",".join(map(str, TRAIN_TPS)))
    p.add_argument("--steps-num", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--per-type-min", type=int, default=40)
    p.add_argument("--top-n", type=int, default=12)
    p.add_argument("--tag", default="latent_grn")
    p.add_argument("--per-type-full-out", default=None)
    p.add_argument("--traj-out", default=None, help="if set, re-simulate forward from real t0 cells and write predicted gene-trajectory")
    p.add_argument("--traj-panel", default="NEUROG3,NEUROD1,NKX2-2,FEV,INS,GCG,SST,SLC30A8")
    args = p.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, ENGINE_SRC)
    sys.path.insert(0, HERE)
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from vae import build_apply
    from grn_drift import GRNDrift
    from datasets import Input_Dataset
    from training_setup import Training_Setup
    from utils import FORWARD

    # ---- frozen VAE + saved standardization ----
    if args.joint_vae:
        from joint_vae import build_apply_joint
        jm = pickle.load(open(args.joint_vae, "rb"))
        jcfg = jm["config"]
        vcfg_j = dict(enc_hidden=jcfg["vae_enc_hidden_dim"], dec_hidden=jcfg["vae_dec_hidden_dim"],
                      latent=jcfg["vae_latent_dim"], n_genes=jcfg["vae_input_dim"],
                      t_dim=jcfg["vae_t_dim"], t_ref=args.t_ref)
        encode_mean, decode, _ = build_apply_joint(jm["vae_params"], vcfg_j)
        vparams = None
        vgenes = None
        latent_dim = int(jcfg["vae_latent_dim"])
    else:
        vae = pickle.load(open(args.vae, "rb"))
        vparams, vcfg, vgenes = vae["params"], vae["cfg"], vae["genes"]
        _, _, encode_mean, decode, _ = build_apply(vcfg)
        latent_dim = vcfg["latent"]

    mp = pickle.load(open(args.model, "rb"))
    z_mean = jnp.asarray(mp["z_mean"]); z_std = jnp.asarray(mp["z_std"])

    # ---- data + labels ----
    df = pd.read_csv(args.data, index_col=0)
    tps = sorted(df["time"].unique()); rev = {k: v for k, v in zip(tps, range(len(tps)))}
    df["time"] = df["time"].map(rev)
    genes = df.columns[:-1].to_numpy()
    if vgenes is not None:
        assert list(genes) == list(vgenes)
    grn = GRNDrift.from_pkl(args.grn_pkl)
    if args.label_mode == "pancreatic":
        meta = pd.read_csv(args.meta, sep="\t", index_col="library.barcode")
        labels = meta["Assigned_cluster"].reindex(df.index)
    else:
        if not args.segment_map:
            raise ValueError("--segment-map is required for --label-mode segment")
        meta = pd.read_csv(args.meta, index_col=0).reindex(df.index)
        sm = pd.read_csv(args.segment_map)
        labels = pd.to_numeric(meta["segment"], errors="coerce").map(
            dict(zip(sm.segment.astype(float), sm.lineage.astype(str))))
    type_ids = grn.labels_to_type_ids(labels.fillna("__unconstrained__").to_numpy())

    # ---- rebuild latent engine (no_vae on latent dim), load forward params ----
    train_tps = [int(t) for t in args.train_tps.split(",")]
    train_df = df[df["time"].isin(train_tps)]
    Ztr = np.asarray(encode_mean(vparams, jnp.asarray(train_df.drop(columns=["time"]).values.astype(np.float32))))
    Ztr_s = (Ztr - np.asarray(z_mean)) / np.asarray(z_std)
    latent_train = pd.DataFrame(Ztr_s, index=train_df.index.to_list()); latent_train["time"] = train_df["time"].values
    ds = Input_Dataset(x=latent_train, meta=None, meta_celltype_column=None,
                       splitting_births_frac=0.9, steps_num=args.steps_num, val_split=False,
                       death_importance_rate=1, f_val=args.f_val, g_type=args.g_type, g_max=args.g_max)
    ts = Training_Setup(dataset=ds, dataset_name="interp", objective="mean_matching", no_vae=True,
                        hidden_dim=[args.hidden_dim] * 2, dec_hidden_size=[args.hidden_dim] * 2 + [1],
                        epochs=1, num_sde=10, paths_reuse=5, reality_coefficient=0.2, ipf_mask_dead=True,
                        t_dim=16, batch_size=args.batch_size, vae_input_dim=latent_dim, vae_latent_dim=latent_dim,
                        vae_enc_hidden_dim=[512, 256, 128], vae_dec_hidden_dim=[512, 256, 128],
                        vae_t_dim=16, vae_batch_size=32, vae_epochs=0, ferryman_hidden_dim=[100] * 3)
    params = dict(ts.state[1]) if ts.state[1] else {}
    for k in ("forward", "backward", "ferryman"):
        if k in mp:
            params[k] = mp[k]
    key = random.PRNGKey(0)
    Z = ts.score(ts.model[FORWARD])

    def gene_drift(zp, t_scalar):
        """dx/dt in gene space for standardized-latent positions zp (n, latent)."""
        zp = jnp.asarray(zp)
        b_lat = ts.sde.f(t_scalar, zp) + ts.sde.g(t_scalar, zp) * Z(params["forward"], key, t_scalar, zp)
        z = zp * z_std + z_mean
        _, dx = jax.jvp(lambda zz: decode(vparams, zz), (z,), (z_std * b_lat,))
        return np.asarray(dx)                         # (n, G)

    # ---- per (type, stage) decoded drift -> CSV ----
    stages = [int(s) for s in args.stages.split(",")]
    rng = np.random.default_rng(0)
    Xall = df.drop(columns=["time"]).values.astype(np.float32)
    time_col = df["time"].values
    full_rows = []
    print(f"{'='*64}\nLATENT-MODEL PER-TYPE DECODED DRIFT  tag={args.tag}\n{'='*64}")
    for tp in stages:
        t_scalar = float(tp) / float(max(df["time"].max(), 1))
        for ty, tid in sorted(grn.type_to_id.items()):
            sel = np.where((time_col == tp) & (type_ids == tid))[0]
            if len(sel) < args.per_type_min:
                continue
            if len(sel) > args.n_cells:
                sel = rng.choice(sel, args.n_cells, replace=False)
            zp = (np.asarray(encode_mean(vparams, jnp.asarray(Xall[sel]))) - np.asarray(z_mean)) / np.asarray(z_std)
            bt = gene_drift(zp, t_scalar).mean(0)
            s = pd.Series(bt, index=genes).sort_values()
            up = ", ".join(f"{g}(+{v:.2f})" for g, v in s[::-1].head(args.top_n).items())
            print(f"[{ty} st{tp} n={len(sel)}] UP: {up}")
            if args.per_type_full_out:
                for g, v in s.items():
                    # Preserve full decoded-drift precision. Early/smoke models can have
                    # magnitudes below 1e-5; rounding here creates massive GSEA rank ties.
                    full_rows.append((args.tag, ty, tp, g, float(v)))
    if args.per_type_full_out and full_rows:
        new = not os.path.exists(args.per_type_full_out)
        with open(args.per_type_full_out, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["tag", "cell_type", "stage", "gene", "drift"])
            w.writerows(full_rows)
        print(f"[per-type-full] appended {len(full_rows)} rows -> {args.per_type_full_out}")

    # ---- ARTEMIS-style trajectory fidelity: re-simulate FORWARD from real t0 cells ----
    if args.traj_out:
        from functools import partial
        from utils import broadcast, FERRYMAN
        gidx = {g: i for i, g in enumerate(list(genes))}
        def resolve(g):
            for c in (g, g.replace("-", "_"), g.replace("_", "-")):
                if c in gidx: return c
            return None
        panel = [(g, resolve(g)) for g in args.traj_panel.split(",")]
        eval_score = broadcast(lambda model, p: partial(ts.score(model), p), ts.model, params)
        eval_ferryman = partial(ts.ferryman.apply, params=params[FERRYMAN], direction=FORWARD)
        t0 = train_tps[0]
        sel0 = np.where(time_col == t0)[0]
        if len(sel0) > 2000:
            sel0 = rng.choice(sel0, 2000, replace=False)
        x_init = (np.asarray(encode_mean(vparams, jnp.asarray(Xall[sel0]))) - np.asarray(z_mean)) / np.asarray(z_std)
        x_init = jnp.asarray(x_init.astype(np.float32))
        steps_num = ds.steps_num
        t_0 = int(ds.cells_time(t0) * steps_num)
        trajs, _, statuses, _ = ts.sde.sample_trajectory(
            random.PRNGKey(0), FORWARD, x_init, eval_score, eval_ferryman, t_0=t_0, corrector="")
        tps_all = sorted(df["time"].unique())
        gt = df.groupby("time")[[c for _, c in panel if c]].mean()
        pred = {}
        for tp in tps_all:
            step = int(ds.cells_time(tp) * steps_num)
            st = np.asarray(statuses[step]); alive = np.where(st)[0]
            lat = np.asarray(trajs[step])[alive] if len(alive) else np.asarray(trajs[step])
            x = np.asarray(decode(vparams, jnp.asarray(lat) * z_std + z_mean))
            pred[tp] = {g: float(x[:, gidx[c]].mean()) for g, c in panel if c}
        preddf = pd.DataFrame(pred).T  # index=tp, cols=gene
        # sanity gate: INS must rise materially across development
        ins_ok = ("INS" in preddf.columns and preddf["INS"].iloc[-1] - preddf["INS"].iloc[0] > 0.2)
        print(f"\n===== TRAJECTORY FIDELITY (re-sim from real t{t0}) — INS sanity gate: "
              f"{'PASS' if ins_ok else 'FAIL (readout NOT trustworthy)'} =====")
        print(f"{'gene':10s} {'Pearson r':>10s} {'GT peak@t':>10s} {'shape':>10s}")
        for g, c in panel:
            if not c: continue
            pv = preddf[g].reindex(tps_all).values.astype(float)
            gv = gt[c].reindex(tps_all).values.astype(float)
            r = np.corrcoef(pv, gv)[0, 1] if pv.std() > 1e-9 and gv.std() > 1e-9 else float("nan")
            pk = int(np.argmax(gv)); shape = "transient" if 0 < pk < len(gv) - 1 else ("rising" if pk == len(gv) - 1 else "falling")
            print(f"{g:10s} {r:>10.3f} {tps_all[pk]:>10} {shape:>10s}")
        for g in ("NEUROG3", "INS"):
            if g in preddf.columns:
                print(f"  {g:8s} GT  : {[round(float(gt[resolve(g)].reindex(tps_all).values[i]),3) for i in range(len(tps_all))]}")
                print(f"  {g:8s} pred: {[round(float(preddf[g].reindex(tps_all).values[i]),3) for i in range(len(tps_all))]}")
        if args.traj_out not in (None, "-"):
            preddf.reset_index().rename(columns={"index":"t"}).to_csv(args.traj_out, index=False)
            print(f"[traj] saved -> {args.traj_out}")
    print("INTERPRET_LATENT_DONE")


if __name__ == "__main__":
    main()
