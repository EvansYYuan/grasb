import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from vae import build_apply


def test_encoder_jvp_has_expected_shapes():
    cfg = {"enc_hidden": [8], "dec_hidden": [8], "latent": 3, "n_genes": 5}
    init, _, encode, decode, enc_jvp = build_apply(cfg)
    x = jnp.ones((2, 5))
    params = init(jax.random.PRNGKey(0), x)
    z, dz = enc_jvp(params, x, jnp.ones_like(x))
    assert z.shape == (2, 3)
    assert dz.shape == (2, 3)
    assert decode(params, encode(params, x)).shape == x.shape

