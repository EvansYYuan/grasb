#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared time-independent VAE for GRASB.

Why a plain VAE and not ARTEMIS's shipped `VariationalAutoEncoder`
(`engine/models.py`)? ARTEMIS's VAE is **time-conditioned** — it concatenates a
timestep embedding onto both the encoder input and the decoder input, so the
latent geometry shifts with developmental time and the pushforward Jacobian would
have to carry t-embedding bookkeeping. For the frozen-denoiser design here we want

    * a single, time-consistent latent manifold across all weeks (standard for
      trajectory inference), and
    * a clean encoder map  z = enc(x)  and decoder  x_hat = dec(z)  so the GRN
      pushforward  J_enc . b_GRN  is a plain jvp with no time term.

So this module defines a minimal x->z->x VAE. It is deliberately isolated under
`latent_grn/` (like `debias_target/`); it shares only the read-only data with the
main pipeline.

The three maps the rest of the experiment needs, all pure functions of the FROZEN
params (so they are safe inside jit and inside the training loss):

    encode_mean(params, x)         -> z            (deterministic latent = posterior mean)
    decode(params, z)              -> x_hat        (gene-space reconstruction)
    enc_jvp(params, x, v)          -> (z, J_enc(x) . v)   (forward-mode; for the pushforward)

`build_apply(cfg)` returns these three closures given a config dict, so
pretrain / encode / train-loss all instantiate the *same* architecture.
"""

import jax
import jax.numpy as jnp
import haiku as hk


# --------------------------------------------------------------------------- net
class _Encoder(hk.Module):
    def __init__(self, hidden, latent, name="enc"):
        super().__init__(name=name)
        self.hidden = hidden
        self.latent = latent

    def __call__(self, x):
        h = x
        for w in self.hidden:
            h = jax.nn.relu(hk.Linear(w)(h))
        mean = hk.Linear(self.latent)(h)
        log_std = hk.Linear(self.latent)(h)
        return mean, log_std


class _Decoder(hk.Module):
    def __init__(self, hidden, out_dim, name="dec"):
        super().__init__(name=name)
        self.hidden = hidden
        self.out_dim = out_dim

    def __call__(self, z):
        h = z
        for w in self.hidden:
            h = jax.nn.relu(hk.Linear(w)(h))
        return hk.Linear(self.out_dim)(h)          # gene-space logits (== recon, Gaussian)


class _VAE(hk.Module):
    def __init__(self, enc_hidden, dec_hidden, latent, out_dim, name="vae"):
        super().__init__(name=name)
        self.encoder = _Encoder(enc_hidden, latent)
        self.decoder = _Decoder(dec_hidden, out_dim)

    def __call__(self, x, key):
        mean, log_std = self.encoder(x)
        std = jnp.exp(log_std)
        z = mean + std * jax.random.normal(key, mean.shape)
        x_hat = self.decoder(z)
        return x_hat, mean, log_std, z


# ------------------------------------------------------------------------- apply
def build_apply(cfg):
    """Return (init, full_apply, encode_mean, decode, enc_jvp) for the given config.

    cfg keys: enc_hidden(list), dec_hidden(list), latent(int), n_genes(int).
    The decoder hidden widths are given ENCODER-order in cfg['dec_hidden'] and
    used as-is (i.e. pass them already reversed if you want a symmetric net).
    """
    enc_hidden = cfg["enc_hidden"]
    dec_hidden = cfg["dec_hidden"]
    latent = cfg["latent"]
    n_genes = cfg["n_genes"]

    def _fwd(x, key):
        return _VAE(enc_hidden, dec_hidden, latent, n_genes)(x, key)

    full = hk.transform(_fwd)

    def _enc_fwd(x):
        return _VAE(enc_hidden, dec_hidden, latent, n_genes).encoder(x)

    def _dec_fwd(z):
        return _VAE(enc_hidden, dec_hidden, latent, n_genes).decoder(z)

    enc_t = hk.transform(_enc_fwd)
    dec_t = hk.transform(_dec_fwd)

    def init(key, x):
        return full.init(key, x, key)

    def full_apply(params, key, x):
        return full.apply(params, key, x, key)

    def encode_mean(params, x):
        mean, _ = enc_t.apply(params, None, x)
        return mean

    def decode(params, z):
        return dec_t.apply(params, None, z)

    def enc_jvp(params, x, v):
        """(z, J_enc(x).v) via forward-mode AD of the deterministic mean encoder."""
        f = lambda xx: enc_t.apply(params, None, xx)[0]
        return jax.jvp(f, (x,), (v,))

    return init, full_apply, encode_mean, decode, enc_jvp
