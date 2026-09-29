#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 1 of GRASB: pretrain and freeze a time-independent VAE.

Trains the `engine/vae.py` VAE on an HVG log1p matrix, restricted
to TRAIN timepoints only (t3/t6 held out) so the frozen representation never sees
the evaluation weeks, preventing leakage into held-out W2 through the decoder.

Outputs `results/vae_frozen.pkl` = {params, cfg, train_r2, val_r2,
gene_names}. Everything downstream (encode, the latent SB, the pushforward GRN
penalty, the decode-for-biology readout) loads THIS frozen artifact.

Run:  OMP_NUM_THREADS=1 python pretrain_vae.py --gpu 0 --epochs 300
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE_SRC = os.path.join(ROOT, "engine")
DEFAULT_DATA = os.path.join(ROOT, "data", "pancreatic_preprocessed.csv")
TRAIN_TPS = [0, 1, 2, 4, 5, 7]
VAL_TPS = [3, 6]


def r2(x, xhat):
    ss_res = np.sum((x - xhat) ** 2)
    ss_tot = np.sum((x - x.mean(0)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--gpu", default="0")
    p.add_argument("--latent", type=int, default=10)
    p.add_argument("--enc-hidden", default="512,256")
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (recon summed over genes dominates).")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "vae_frozen.pkl"))
    p.add_argument("--train-tps", default=",".join(map(str, TRAIN_TPS)))
    p.add_argument("--val-tps", default=",".join(map(str, VAL_TPS)))
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, ENGINE_SRC)

    import jax
    import jax.numpy as jnp
    import optax
    from vae import build_apply
    print(f"[env] jax {jax.__version__} devices={jax.devices()}", flush=True)

    # ---- data (train cells only for the VAE fit) ----
    df = pd.read_csv(args.data, index_col=0)
    tps = sorted(df["time"].unique())
    rev = {k: v for k, v in zip(tps, np.arange(len(tps)))}
    df["time"] = df["time"].map(rev)
    train_tps = [int(t) for t in args.train_tps.split(",")]
    val_tps = [int(t) for t in args.val_tps.split(",")]
    leak = sorted(set(train_tps) & set(val_tps))
    if leak:
        raise ValueError(f"Train/validation timepoints overlap: {leak}")
    genes = df.drop(columns=["time"]).columns.to_numpy()
    Xtr = df[df["time"].isin(train_tps)].drop(columns=["time"]).values.astype(np.float32)
    Xval = df[df["time"].isin(val_tps)].drop(columns=["time"]).values.astype(np.float32)
    n_genes = Xtr.shape[1]
    print(f"[data] train {Xtr.shape} | val {Xval.shape} | genes {n_genes}", flush=True)

    enc_hidden = [int(x) for x in args.enc_hidden.split(",")]
    dec_hidden = enc_hidden[::-1]                       # symmetric decoder
    cfg = dict(enc_hidden=enc_hidden, dec_hidden=dec_hidden, latent=args.latent,
               n_genes=n_genes, beta=args.beta)
    init, full_apply, encode_mean, decode, _ = build_apply(cfg)

    key = jax.random.PRNGKey(args.seed)
    key, ik = jax.random.split(key)
    params = init(ik, jnp.asarray(Xtr[:args.batch]))
    n_params = sum(int(np.prod(v.shape)) for m in params.values() for v in m.values())
    print(f"[model] enc_hidden={enc_hidden} latent={args.latent} params={n_params}", flush=True)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    def elbo(params, key, x):
        # deterministic-mean encoder + reparam sample; Gaussian (MSE) reconstruction.
        x_hat, mean, log_std, z = full_apply(params, key, x)
        recon = jnp.sum(jnp.square(x - x_hat), axis=-1)                 # sum over genes
        var = jnp.exp(2.0 * log_std)
        kl = -0.5 * jnp.sum(1.0 + 2.0 * log_std - jnp.square(mean) - var, axis=-1)
        return jnp.mean(recon + args.beta * kl), (jnp.mean(recon), jnp.mean(kl))

    @jax.jit
    def step(params, opt_state, key, x):
        (loss, aux), grads = jax.value_and_grad(elbo, has_aux=True)(params, key, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    Xtr_j = jnp.asarray(Xtr)
    n = Xtr.shape[0]
    steps_per_epoch = max(1, n // args.batch)
    for ep in range(args.epochs):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, n)
        ep_recon = ep_kl = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            key, stk = jax.random.split(key)
            params, opt_state, loss, aux = step(params, opt_state, stk, Xtr_j[idx])
            ep_recon += float(aux[0]); ep_kl += float(aux[1])
        if ep % 20 == 0 or ep == args.epochs - 1:
            xhat_tr = np.asarray(decode(params, encode_mean(params, Xtr_j)))
            print(f"[ep {ep:4d}] recon/cell {ep_recon/steps_per_epoch:8.2f} "
                  f"kl/cell {ep_kl/steps_per_epoch:6.2f} | train R2 {r2(Xtr, xhat_tr):.4f}", flush=True)

    # ---- frozen readout: R2 on train and (unseen) held-out cells ----
    xhat_tr = np.asarray(decode(params, encode_mean(params, jnp.asarray(Xtr))))
    xhat_val = np.asarray(decode(params, encode_mean(params, jnp.asarray(Xval))))
    train_r2, val_r2 = r2(Xtr, xhat_tr), r2(Xval, xhat_val)
    zt = np.asarray(encode_mean(params, jnp.asarray(Xtr)))
    print(f"\n[frozen] train R2 {train_r2:.4f} | held-out(t3,t6) R2 {val_r2:.4f}", flush=True)
    print(f"[latent] z per-dim std: {np.array2string(zt.std(0), precision=3)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(params=jax.device_get(params), cfg=cfg, genes=genes,
                         train_r2=float(train_r2), val_r2=float(val_r2),
                         latent_std=zt.std(0)), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[save] {args.out}", flush=True)
    print("PRETRAIN_VAE_DONE", flush=True)


if __name__ == "__main__":
    main()
