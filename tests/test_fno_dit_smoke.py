import jax
import jax.numpy as jnp

from models.dit import DiT
from models.fno import FNO1d, FNO2d


def test_fno1d_forward_shape():
    model = FNO1d(num_channels=1)
    x = jnp.ones((2, 16, 1))
    t = jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape


def test_fno2d_forward_shape():
    model = FNO2d(num_channels=1)
    x = jnp.ones((2, 8, 8, 1))
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)
    assert y.shape == x.shape


def test_dit_forward_shape():
    model = DiT(hidden_size=64, depth=2, num_heads=4, patch_size=2, out_channels=1)
    x = jnp.ones((2, 8, 8, 1))
    t = jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape
