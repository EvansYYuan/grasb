#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 2 of GRASB: train a latent Schrodinger bridge with optional GRN guidance.

Pipeline (frozen arm):
  1. load a frozen VAE (results/vae_frozen.pkl)
  2. encode train+val gene matrices -> latent z = enc_mean(x); standardize with
     TRAIN (z_mean, z_std) so the SB runs in ~unit-scale latent
  3. run the existing no_vae engine on the 10-d latent vectors (SB is space-agnostic)
  4. if --lambda-grn>0: attach the gene-space GRN with grn_latent=True; the loss
     decodes each latent trajectory pos, evaluates gene-space b_GRN, and pushes it
     into standardized-latent units via J_enc (closures built here, frozen params)
  5. predict held-out t3/t6: simulate in latent, DECODE to gene space, W2 vs gene GT

With ``--lambda-grn 0`` this is the matched latent no-GRN baseline.

Run:  OMP_NUM_THREADS=1 python train_latent_grn.py --gpu 0 --g-max 0.5 --save \
        [--lambda-grn 1 --grn-pkl ../output/pancreatic_b_grn_...pkl --grn-target regulon]
"""
import argparse
import os
import sys
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE_SRC = os.path.join(ROOT, "engine")
DEFAULT_DATA = os.path.join(ROOT, "data", "pancreatic_preprocessed.csv")
DEFAULT_VAE = os.path.join(ROOT, "results", "vae_frozen.pkl")
TRAIN_TPS = [0, 1, 2, 4, 5, 7]
VAL_TPS = [3, 6]


def latent_df(z, index, time):
    df = pd.DataFrame(z, index=list(index))
    df["time"] = np.asarray(time)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--joint-vae", default=None,
                   help="If set, FREEZE the jointly-trained ARTEMIS-style VAE from this engine "
                        "model_params pkl and use it (read at --t-ref) instead of the frozen plain "
                        "VAE. Isolates the representation (joint vs plain weights) as the only change.")
    p.add_argument("--t-ref", type=float, default=3.5,
                   help="Reference time (VAE integer-stage units 0..7) at which the t-conditioned "
                        "joint VAE is frozen into a single chart. Used for encode+pushforward+decode.")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    p.add_argument("--gpu", default="0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--num-sde", type=int, default=10, help="IPF outer iterations (bridge convergence)")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--f-val", type=float, default=0.0)
    p.add_argument("--g-type", default="triangular", choices=["triangular", "constant", "decreasing", "inverse_triangular"])
    p.add_argument("--g-max", type=float, default=0.5, help="latent SDE diffusion peak (latent std~1 => ~0.5).")
    p.add_argument("--objective", default="mean_matching",
                   choices=["mean_matching", "divergence", "combined"],
                   help="IPF objective. mean_matching was forced by gene-space dim; in 10-d latent "
                        "the divergence (Hutchinson-trace) objective is cheap again. mean_matching "
                        "was FORCED in gene space (trace intractable at G~2000); in latent we can "
                        "test divergence too. GRN penalty wired into mean_matching AND divergence "
                        "(each against its OWN drift convention; see engine/loss.py _grn_penalty).")
    p.add_argument("--lambda-grn", type=float, default=0.0)
    p.add_argument("--grn-pkl", default=os.path.join(
        ROOT, "results", "pancreatic_grn.pkl"))
    p.add_argument("--grn-target", choices=["full", "regulon"], default="regulon")
    p.add_argument("--grn-penalty", choices=["l2", "cosine"], default="l2",
                   help="FORM of the GRN leg. l2 = ||b_θ − b_GRN||² (pins direction AND "
                        "magnitude; the whole +2.43 W2). cosine = 1 − cos(b_θ, b_GRN): steers "
                        "DIRECTION only and leaves ||b_θ|| to the transport objective.")
    p.add_argument("--meta", default=os.path.join(
        ROOT, "data", "pancreatic_metadata.tsv"))
    p.add_argument("--label-mode", choices=["pancreatic", "segment"], default="pancreatic")
    p.add_argument("--segment-map", default=None,
                   help="CSV with segment,lineage columns; required for --label-mode segment.")
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed for the SB trainer (noise-bar repeats).")
    p.add_argument("--train-tps", default="0,1,2,4,5,7",
                   help="TRAIN weeks the bridge may see (VAL_TPS 3,6 always held out). "
                        "Sparser => the bridge must span a LONGER delta-t (delta-t probe).")
    p.add_argument("--run-tag", default="")
    p.add_argument("--val-tps", default="3,6")
    p.add_argument("--dataset-name", default="pancreatic_latent_grn")
    p.add_argument("--output-prefix", default="pancreatic")
    p.add_argument("--steps-num", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--td-sched", type=int, default=6)
    args = p.parse_args()

    global TRAIN_TPS, VAL_TPS
    TRAIN_TPS = [int(t) for t in args.train_tps.split(",")]
    VAL_TPS = [int(t) for t in args.val_tps.split(",")]
    leak = sorted(set(TRAIN_TPS) & set(VAL_TPS))
    if leak:
        raise SystemExit(f"[leak] --train-tps contains held-out week(s) {leak}")
    print(f"[tps] train={TRAIN_TPS} val={VAL_TPS}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, ENGINE_SRC)
    sys.path.insert(0, HERE)

    import jax
    import jax.numpy as jnp
    import jax.random as random
    from vae import build_apply
    from datasets import Input_Dataset
    from training_setup import Training_Setup
    from training import Trainer
    from analysis_utils import get_identity_latents, get_predictions, get_metrics
    print(f"[env] latent-GRN engine gpu={args.gpu} jax devices={jax.devices()}", flush=True)

    # ---------- frozen VAE (plain, time-consistent) OR frozen JOINT VAE (t-conditioned,
    #            read at a single reference time so it becomes one chart) ----------
    if args.joint_vae:
        from joint_vae import build_apply_joint
        jm = pickle.load(open(args.joint_vae, "rb"))
        jcfg = jm["config"]
        vcfg_j = dict(enc_hidden=jcfg["vae_enc_hidden_dim"], dec_hidden=jcfg["vae_dec_hidden_dim"],
                      latent=jcfg["vae_latent_dim"], n_genes=jcfg["vae_input_dim"],
                      t_dim=jcfg["vae_t_dim"], t_ref=args.t_ref)
        encode_mean, decode, enc_jvp = build_apply_joint(jm["vae_params"], vcfg_j)
        vparams = None                      # engine params are baked into the closures
        vgenes = None                       # joint VAE trained on the same csv gene order
        latent_dim = int(jcfg["vae_latent_dim"])
        print(f"[vae] JOINT (frozen) latent={latent_dim} t_ref={args.t_ref} src={os.path.basename(args.joint_vae)}", flush=True)
    else:
        with open(args.vae, "rb") as f:
            vae = pickle.load(f)
        vparams, vcfg, vgenes = vae["params"], vae["cfg"], vae["genes"]
        _, _, encode_mean, decode, enc_jvp = build_apply(vcfg)
        latent_dim = vcfg["latent"]
        print(f"[vae] plain (frozen) latent={latent_dim} train_R2={vae['train_r2']:.4f} val_R2={vae['val_r2']:.4f}", flush=True)

    # ---------- data ----------
    df = pd.read_csv(args.data, index_col=0)
    tps = sorted(df["time"].unique())
    rev = {k: v for k, v in zip(tps, np.arange(len(tps)))}
    df["time"] = df["time"].map(rev)
    tps_sorted = sorted(df["time"].unique().tolist())
    genes = df.drop(columns=["time"]).columns.to_numpy()
    if vgenes is not None:
        assert list(genes) == list(vgenes), "gene order mismatch between data and frozen VAE"

    train_df = df[df["time"].isin(TRAIN_TPS)]
    val_df = df[df["time"].isin(VAL_TPS)]
    Xtr = jnp.asarray(train_df.drop(columns=["time"]).values.astype(np.float32))
    Xval = jnp.asarray(val_df.drop(columns=["time"]).values.astype(np.float32))

    # ---------- encode -> standardized latent ----------
    Ztr = np.asarray(encode_mean(vparams, Xtr))
    Zval = np.asarray(encode_mean(vparams, Xval))
    z_mean = Ztr.mean(0).astype(np.float32)
    z_std = (Ztr.std(0) + 1e-6).astype(np.float32)
    Ztr_s = (Ztr - z_mean) / z_std
    Zval_s = (Zval - z_mean) / z_std
    z_mean_j, z_std_j = jnp.asarray(z_mean), jnp.asarray(z_std)
    print(f"[latent] std-latent train {Ztr_s.shape} | z_std raw {np.array2string(z_std, precision=2)}", flush=True)

    train_latent_data = latent_df(Ztr_s, train_df.index, train_df["time"].values)
    val_latent_data = latent_df(Zval_s, val_df.index, val_df["time"].values)

    # ---------- GRN (gene space) + pushforward closures ----------
    grn = None
    cell_type_ids = None
    grn_dec_fn = grn_encjvp_fn = None
    if args.lambda_grn > 0:
        from grn_drift import GRNDrift
        grn = GRNDrift.from_pkl(args.grn_pkl)
        grn.component = args.grn_target
        grn_gene_norm = [str(g).upper() for g in grn.genes]
        data_gene_norm = [str(g).upper() for g in genes]
        assert grn_gene_norm == data_gene_norm, "GRN gene order mismatch after case normalization"
        if args.label_mode == "pancreatic":
            meta = pd.read_csv(args.meta, sep="\t", index_col="library.barcode")
            ct = meta["Assigned_cluster"].reindex(train_df.index)
        else:
            if not args.segment_map:
                raise ValueError("--segment-map is required for --label-mode segment")
            meta = pd.read_csv(args.meta, index_col=0).reindex(train_df.index)
            sm = pd.read_csv(args.segment_map)
            seg_to_lineage = dict(zip(sm.segment.astype(float), sm.lineage.astype(str)))
            ct = pd.to_numeric(meta["segment"], errors="coerce").map(seg_to_lineage)
        # Missing/non-selected terminal segments are intentionally unconstrained.
        ct = ct.fillna("__unconstrained__")
        cell_type_ids = grn.labels_to_type_ids(ct.to_numpy())
        # closures in STANDARDIZED latent: dec(z') = decoder(z'*z_std+z_mean);
        # encjvp(x,v) = J_enc(x).v / z_std  (gene-velocity -> std-latent velocity).
        def grn_dec_fn(zp):
            return decode(vparams, zp * z_std_j + z_mean_j)
        def grn_encjvp_fn(x, v):
            _, jv = enc_jvp(vparams, x, v)
            return jv / z_std_j
        print(f"[grn] latent-pushforward λ={args.lambda_grn} target={args.grn_target} "
              f"penalty={args.grn_penalty} pkl={os.path.basename(args.grn_pkl)} "
              f"bins={grn.n_bins} labeled={len(cell_type_ids)} "
              f"unknown={int((cell_type_ids<0).sum())}", flush=True)

    # ---------- dataset / setup / trainer (SB in latent, no_vae engine) ----------
    input_dim = latent_dim
    enc_hidden = [args.hidden_dim] * 2
    dec_hidden = [args.hidden_dim] * 2 + [1]        # scalar-potential head (Z = g·∇φ)

    train_dataset = Input_Dataset(
        x=train_latent_data, meta=None, meta_celltype_column=None,
        splitting_births_frac=0.9, steps_num=args.steps_num, val_split=False,
        death_importance_rate=1, f_val=args.f_val, g_type=args.g_type, g_max=args.g_max,
        cell_type_ids=cell_type_ids)

    ts = Training_Setup(
        dataset=train_dataset, dataset_name=args.dataset_name,
        objective=args.objective, no_vae=True,
        hidden_dim=enc_hidden, dec_hidden_size=dec_hidden,
        epochs=args.epochs, num_sde=args.num_sde, paths_reuse=5,
        reality_coefficient=0.2, ipf_mask_dead=True, t_dim=16, batch_size=args.batch_size,
        vae_input_dim=input_dim, vae_latent_dim=input_dim,
        vae_enc_hidden_dim=[512, 256, 128], vae_dec_hidden_dim=[512, 256, 128],
        vae_t_dim=16, vae_batch_size=32, vae_epochs=0,
        ferryman_hidden_dim=[100] * 3,
        grn=grn, lambda_grn=args.lambda_grn, grn_penalty=args.grn_penalty,
        grn_latent=(grn is not None), grn_dec_fn=grn_dec_fn, grn_encjvp_fn=grn_encjvp_fn)

    tr = Trainer(dataset=train_dataset, ts=ts, key=random.PRNGKey(args.seed),
                 lr=1e-4, vae_lr=1e-4, ferryman_lr=1e-4, ferryman_coeff=1)
    print(f"[sde] f_val={args.f_val} g_type={args.g_type} g_max={args.g_max} "
          f"lambda_grn={args.lambda_grn}", flush=True)

    print("[train] starting latent SB ...", flush=True)
    tr_model = tr.train(td_schedule=[1] * args.td_sched, project_name=args.dataset_name)

    # ---------- predict in latent, DECODE, W2 in gene space ----------
    train_recon, train_latent, val_recon, val_latent = get_identity_latents(
        train_latent_data, val_latent_data)
    predictions, _ = get_predictions(
        train_latent_data, val_latent_data, TRAIN_TPS, VAL_TPS, train_dataset, train_latent,
        tps_sorted, tr_model, ts, input_dim, None, input_dim,
        t_0_orig=TRAIN_TPS[0], no_vae=True)
    sims = predictions["simulations"]

    def decode_sim(latent_w_time, t):
        sub = latent_w_time[latent_w_time["time"] == t]
        zp = jnp.asarray(sub.values[:, :-1].astype(np.float32))
        return np.asarray(decode(vparams, zp * z_std_j + z_mean_j))

    print("\n===== W2 ON HELD-OUT TIMEPOINTS (decoded to gene space) =====", flush=True)
    w2s = {}
    for t in VAL_TPS:
        gt = val_df[val_df["time"] == t].values[:, :-1]
        perf = []
        for i in range(len(sims)):
            pred_gene = decode_sim(sims[i][0], t)            # [0]=latent+time (identity in no_vae)
            # reuse compute_metrics_subset via a tiny sim dict
            from evaluate import compute_metrics_subset
            perf.append(compute_metrics_subset(gt, pred_gene)["w2"])
        w2s[t] = (float(np.mean(perf)), float(np.std(perf)))
        print(f"W2  t={t} : {w2s[t][0]:.4f} +- {w2s[t][1]:.4f}", flush=True)

    if args.save:
        os.makedirs(args.out_dir, exist_ok=True)
        pfx = "jointfrozen" if args.joint_vae else "latent"
        tag = f"{pfx}_{args.objective}_g{args.g_max:g}_lam{args.lambda_grn:g}"
        if args.lambda_grn > 0 and args.grn_target != "full":
            tag += f"_{args.grn_target}"
        if args.lambda_grn > 0 and args.grn_penalty != "l2":
            tag += f"_{args.grn_penalty}"
        if args.run_tag:
            tag += f"_{args.run_tag}"
        model_params = {
            "forward": tr_model.training_setup.state[1]["forward"],
            "backward": tr_model.training_setup.state[1]["backward"],
            "ferryman": tr_model.training_setup.state[1]["ferryman"],
            "config": tr_model.get_model_configs(),
            "z_mean": z_mean, "z_std": z_std, "vae": args.vae, "w2": w2s,
            "joint_vae": args.joint_vae, "t_ref": args.t_ref,
        }
        with open(os.path.join(args.out_dir, f"{args.output_prefix}_{tag}_model_params.pkl"), "wb") as f:
            pickle.dump(model_params, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(args.out_dir, f"{args.output_prefix}_{tag}_predictions.pkl"), "wb") as f:
            pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] wrote model params + predictions ({tag})", flush=True)

    print("TRAIN_LATENT_DONE", flush=True)


if __name__ == "__main__":
    main()
