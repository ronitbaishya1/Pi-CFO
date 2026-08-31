import jax
import jax.numpy as jnp

from utils.physics_swe import (
    swe_residual,
    swe_physics_loss,
)


# ============================================================
# TEST 1
# Still, spatially uniform water
# ============================================================

def test_still_uniform_water():

    batch_size = 2
    nx = 32
    ny = 32

    # q[..., 0] = h
    # q[..., 1] = hu
    # q[..., 2] = hv

    q = jnp.zeros(
        (batch_size, nx, ny, 3),
        dtype=jnp.float32,
    )

    # Uniform water depth
    h = 1.0

    q = q.at[..., 0].set(h)

    # Still water:
    # hu = 0
    # hv = 0
    #
    # and therefore q_t = 0
    q_t = jnp.zeros_like(q)

    residual = swe_residual(
        q=q,
        q_t=q_t,
        dx=0.15625,
        dy=0.15625,
        g=1.0,
    )

    max_residual = jnp.max(
        jnp.abs(residual)
    )

    print(
        "Still-water maximum residual:",
        max_residual,
    )

    assert max_residual < 1e-6


# ============================================================
# TEST 2
# Spatially uniform moving water
# ============================================================

def test_uniform_moving_water():

    batch_size = 2
    nx = 32
    ny = 32

    q = jnp.zeros(
        (batch_size, nx, ny, 3),
        dtype=jnp.float32,
    )

    # Constant depth
    q = q.at[..., 0].set(1.5)

    # Constant x momentum
    q = q.at[..., 1].set(0.3)

    # Constant y momentum
    q = q.at[..., 2].set(-0.2)

    # Because every field is spatially uniform,
    # all spatial derivatives are zero.
    #
    # Therefore this can be a constant state:
    q_t = jnp.zeros_like(q)

    residual = swe_residual(
        q=q,
        q_t=q_t,
        dx=0.15625,
        dy=0.15625,
        g=1.0,
    )

    max_residual = jnp.max(
        jnp.abs(residual)
    )

    print(
        "Uniform-flow maximum residual:",
        max_residual,
    )

    assert max_residual < 1e-6


# ============================================================
# TEST 3
# JAX can differentiate through the SWE physics loss
# ============================================================

def test_physics_loss_gradient():

    nx = 32
    ny = 32

    q = jnp.zeros(
        (1, nx, ny, 3),
        dtype=jnp.float32,
    )

    # Uniform water depth
    q = q.at[..., 0].set(1.0)

    # Start with zero predicted time derivative
    q_t = jnp.zeros_like(q)

    def loss_from_qt(q_t_input):

        return swe_physics_loss(
            q=q,
            q_t=q_t_input,
            dx=0.15625,
            dy=0.15625,
            g=1.0,
        )

    grads = jax.grad(
        loss_from_qt
    )(q_t)

    print(
        "Gradient finite:",
        jnp.all(jnp.isfinite(grads)),
    )

    assert jnp.all(
        jnp.isfinite(grads)
    )