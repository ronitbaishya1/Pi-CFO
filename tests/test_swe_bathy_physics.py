"""Phase 52 tests for the bathymetry SWE physics residual."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from utils.physics_swe import swe_residual as flat_swe_residual
from utils.physics_swe_bathy import (
    lake_at_rest_state,
    swe_bathy_physics_loss,
    swe_bathy_residual,
)


def test_zero_bathymetry_matches_flatbed_residual():
    """
    If b = 0 everywhere, the new bathymetry SWE residual should
    reduce to the original flat-bed SWE residual.
    """

    rng = np.random.default_rng(123)

    q = jnp.asarray(
        rng.normal(size=(2, 10, 12, 3)),
        dtype=jnp.float32,
    )

    # Make sure water depth is positive.
    q = q.at[..., 0].set(
        jnp.abs(q[..., 0]) + 1.0
    )

    q_t = jnp.asarray(
        rng.normal(size=q.shape),
        dtype=jnp.float32,
    )

    b = jnp.zeros(
        (2, 10, 12, 1),
        dtype=jnp.float32,
    )

    r_old = flat_swe_residual(
        q,
        q_t,
        dx=0.2,
        dy=0.3,
        g=1.0,
    )

    r_new = swe_bathy_residual(
        q,
        q_t,
        b,
        dx=0.2,
        dy=0.3,
        g=1.0,
    )

    np.testing.assert_allclose(
        np.asarray(r_new),
        np.asarray(r_old),
        rtol=1e-6,
        atol=1e-6,
    )


def test_lake_at_rest_residual_is_near_zero():
    """
    A lake-at-rest state should satisfy the new discrete
    bathymetry SWE residual.

    eta = h + b = constant
    hu = 0
    hv = 0
    q_t = 0
    """

    nx = 16
    ny = 16

    x = jnp.linspace(
        -1.0,
        1.0,
        nx,
    )

    y = jnp.linspace(
        -1.0,
        1.0,
        ny,
    )

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij",
    )

    # Gaussian hill
    b2d = (
        0.2
        *
        jnp.exp(
            -0.5
            *
            (
                (X / 0.4) ** 2
                +
                (Y / 0.4) ** 2
            )
        )
    )

    b = b2d[
        None,
        ...,
        None,
    ]

    q = lake_at_rest_state(
        b,
        eta0=2.0,
    )

    q_t = jnp.zeros_like(
        q
    )

    dx = float(
        x[1] - x[0]
    )

    dy = float(
        y[1] - y[0]
    )

    residual = swe_bathy_residual(
        q,
        q_t,
        b,
        dx=dx,
        dy=dy,
        g=1.0,
    )

    max_residual = float(
        jnp.max(
            jnp.abs(
                residual
            )
        )
    )

    print(
        "\nLake-at-rest max residual:",
        max_residual,
    )

    assert (
        max_residual
        <
        2e-5
    )


def test_bathymetry_physics_loss_has_finite_gradient():
    """
    JAX must be able to differentiate the bathymetry physics
    loss with respect to q_t.
    """

    nx = 12
    ny = 12

    q = jnp.ones(
        (1, nx, ny, 3),
        dtype=jnp.float32,
    )

    q = q.at[
        ...,
        1:
    ].set(
        0.05
    )

    x = jnp.linspace(
        -1.0,
        1.0,
        nx,
    )

    y = jnp.linspace(
        -1.0,
        1.0,
        ny,
    )

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij",
    )

    b = (
        0.1
        *
        jnp.exp(
            -0.5
            *
            (
                (X / 0.5) ** 2
                +
                (Y / 0.5) ** 2
            )
        )
    )[
        None,
        ...,
        None,
    ]

    def loss_fn(q_t):

        return swe_bathy_physics_loss(
            q,
            q_t,
            b,
            dx=2.0 / (nx - 1),
            dy=2.0 / (ny - 1),
            g=1.0,
        )

    q_t = jnp.zeros_like(
        q
    )

    grad = jax.grad(
        loss_fn
    )(
        q_t
    )

    assert jnp.all(
        jnp.isfinite(
            grad
        )
    )