import jax
import jax.numpy as jnp

from models.mlp import SimpleMLP
from models.unet import UNet1D, UNet2D


def test_unet1d_init_and_forward():
    model = UNet1D(use_condition=False)
    x = jnp.ones((2, 16))
    t = jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape


def test_unet2d_init_and_forward():
    model = UNet2D(use_condition=False)
    x = jnp.ones((2, 8, 8, 1))
    t = jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape


def test_simple_mlp_forward_shape():
    model = SimpleMLP(output_dim=3)
    x = jnp.ones((4, 3))
    t = jnp.ones((4,))
    params = model.init(jax.random.PRNGKey(0), x, t)
    y = model.apply(params, x, t)
    assert y.shape == x.shape
