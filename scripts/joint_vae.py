#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Adapter that exposes the engine's *jointly-trained* (ARTEMIS-style) VAE with the
SAME three closures the frozen-plain-VAE experiment uses (vae.build_apply), so the
latent-GRN pipeline (train_latent_grn.py / interpret_latent.py) runs on the joint
representation with a one-line swap.

Why an adapter is needed
------------------------
The engine VAE (models.VariationalAutoEncoder) is **time-conditioned**: it
concatenates a timestep embedding get_timestep_embedding(t) onto BOTH the encoder
input and the decoder input. So encode/decode are a *family* of maps indexed by t,
not a single chart.

The latent-GRN penalty  ||b_theta(z) - J_enc . b_GRN(D(z))||^2  is only well posed
if the map used to BUILD the SB latents (encode) and the map used in the pushforward
(J_enc, D) are the SAME chart -- otherwise b_theta and J_enc.b_GRN live in different
latent frames. We therefore FREEZE the jointly-trained VAE and read it at a single
reference time  t_ref  (in the VAE's native integer-stage units, 0..7). This turns
the t-conditioned VAE into one time-consistent chart, exactly analogous to the
frozen plain VAE -- so the ONLY thing that differs from the validated frozen-plain
arm is the *representation* (joint-trained weights vs plain weights). That isolation
is the whole point of the joint arm.

`build_apply_joint(vae_params, cfg)` returns
    (encode_mean, decode, enc_jvp)
with vae.py-compatible signatures  fn(params, ...)  (the leading `params` arg is
ignored; the frozen engine params are baked into the closures), so callers can do:

    encode_mean, decode, enc_jvp = build_apply_joint(vae_params, cfg)
    z   = encode_mean(None, x)
    xhat= decode(None, z)
    _, jv = enc_jvp(None, x, v)

cfg keys: enc_hidden(list, engine order e.g. [512,256,128]), dec_hidden(list, same),
latent(int), n_genes(int), t_dim(int), t_ref(float, integer-stage units).
"""
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from models import Encoder, Decoder, get_timestep_embedding


def _split_params(vae_params):
    """Strip the 'variational_auto_encoder/~/' prefix so params key standalone
    Encoder/Decoder modules ('encoder/linear...', 'decoder/linear...')."""
    enc_p, dec_p = {}, {}
    for k, v in vae_params.items():
        short = k.split("~/")[-1]              # e.g. 'encoder/linear_1'
        if short.startswith("encoder"):
            enc_p[short] = v
        elif short.startswith("decoder"):
            dec_p[short] = v
        else:
            raise KeyError(f"unexpected VAE param module: {k}")
    return enc_p, dec_p


def build_apply_joint(vae_params, cfg):
    enc_hidden = cfg["enc_hidden"]
    dec_hidden = cfg["dec_hidden"]
    latent = int(cfg["latent"])
    n_genes = int(cfg["n_genes"])
    t_dim = int(cfg["t_dim"])
    t_ref = float(cfg["t_ref"])

    # constant reference-time embedding (1, t_dim); broadcast per batch
    temb = np.asarray(get_timestep_embedding(jnp.asarray([t_ref], dtype=jnp.float32), t_dim))
    temb_j = jnp.asarray(temb)

    enc_t = hk.transform(lambda xt: Encoder(enc_hidden, latent)(xt))
    dec_t = hk.transform(lambda zt: Decoder(n_genes, dec_hidden)(zt))

    enc_p, dec_p = _split_params(vae_params)

    def _cat_t(a):
        return jnp.concatenate([a, jnp.broadcast_to(temb_j, (a.shape[0], t_dim))], axis=-1)

    def encode_mean(params, x):
        mean, _ = enc_t.apply(enc_p, None, _cat_t(jnp.asarray(x)))
        return mean

    def decode(params, z):
        return dec_t.apply(dec_p, None, _cat_t(jnp.asarray(z)))

    def enc_jvp(params, x, v):
        f = lambda xx: enc_t.apply(enc_p, None, _cat_t(xx))[0]
        return jax.jvp(f, (jnp.asarray(x),), (jnp.asarray(v),))

    return encode_mean, decode, enc_jvp
