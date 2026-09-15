"""Phases 51-52: well-balanced bathymetry residual for 2D SWE.

State
-----
q = [h, hu, hv]

Continuous equations
--------------------
h_t  + (hu)_x + (hv)_y = 0

(hu)_t
+ (hu^2/h + 0.5*g*h^2)_x
+ (hu*hv/h)_y
= -g*h*b_x

(hv)_t
+ (hu*hv/h)_x
+ (hv^2/h + 0.5*g*h^2)_y
= -g*h*b_y

Discrete well-balancing
-----------------------
A naive centered discretization of g*h*b_x does not exactly cancel the
centered pressure-gradient term at lake-at-rest. Here the source uses a
neighbor-averaged depth:

    hbar_x = 0.5 * (h_{i+1}+h_{i-1})
    Sx     = g*hbar_x*(b_{i+1}-b_{i-1})/(2*dx)

At eta=h+b=constant, b_{i+1}-b_{i-1}=-(h_{i+1}-h_{i-1}), so Sx exactly
cancels the centered derivative of 0.5*g*h^2. The same is done in y.

This is the training/evaluation residual discretization. The PyClaw dataset
generator independently uses Clawpack's f-wave well-balanced solver.
"""

from __future__ import annotations

import jax.numpy as jnp


def _safe_depth(h: jnp.ndarray, floor: float = 1e-6) -> jnp.ndarray:
    return jnp.maximum(h, floor)


def swe_bathy_fluxes(
    q: jnp.ndarray,
    g: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    h = q[..., 0]
    hu = q[..., 1]
    hv = q[..., 2]
    h_safe = _safe_depth(h)

    fx = jnp.stack(
        [
            hu,
            hu**2 / h_safe + 0.5 * g * h**2,
            hu * hv / h_safe,
        ],
        axis=-1,
    )

    fy = jnp.stack(
        [
            hv,
            hu * hv / h_safe,
            hv**2 / h_safe + 0.5 * g * h**2,
        ],
        axis=-1,
    )

    return fx, fy


def ddx_center(f: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Centered x derivative, returning the common interior."""
    return (
        f[:, 2:, 1:-1, :]
        - f[:, :-2, 1:-1, :]
    ) / (2.0 * dx)


def ddy_center(f: jnp.ndarray, dy: float) -> jnp.ndarray:
    """Centered y derivative, returning the common interior."""
    return (
        f[:, 1:-1, 2:, :]
        - f[:, 1:-1, :-2, :]
    ) / (2.0 * dy)


def well_balanced_bed_source(
    q: jnp.ndarray,
    bathymetry: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    g: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return +g*h*b_x and +g*h*b_y on the common interior.

    These are LHS source contributions. The continuous momentum equations
    have -g*h*b_x/-g*h*b_y on the RHS.
    """
    h = q[..., 0]

    if bathymetry.ndim == 4:
        b = bathymetry[..., 0]
    elif bathymetry.ndim == 3:
        b = bathymetry
    else:
        raise ValueError(
            "bathymetry must have shape (B,H,W) or (B,H,W,1)."
        )

    hbar_x = 0.5 * (
        h[:, 2:, 1:-1]
        + h[:, :-2, 1:-1]
    )
    dbdx = (
        b[:, 2:, 1:-1]
        - b[:, :-2, 1:-1]
    ) / (2.0 * dx)

    hbar_y = 0.5 * (
        h[:, 1:-1, 2:]
        + h[:, 1:-1, :-2]
    )
    dbdy = (
        b[:, 1:-1, 2:]
        - b[:, 1:-1, :-2]
    ) / (2.0 * dy)

    source_x = g * hbar_x * dbdx
    source_y = g * hbar_y * dbdy

    return source_x, source_y


def swe_bathy_residual(
    q: jnp.ndarray,
    q_t: jnp.ndarray,
    bathymetry: jnp.ndarray,
    *,
    dx: float = 0.15625,
    dy: float = 0.15625,
    g: float = 1.0,
) -> jnp.ndarray:
    """Return [R_h,R_hu,R_hv] on interior cells."""
    fx, fy = swe_bathy_fluxes(q, g=g)

    dfx = ddx_center(fx, dx)
    dfy = ddy_center(fy, dy)

    q_t_interior = q_t[:, 1:-1, 1:-1, :]

    source_x, source_y = well_balanced_bed_source(
        q,
        bathymetry,
        dx=dx,
        dy=dy,
        g=g,
    )

    r_h = (
        q_t_interior[..., 0]
        + dfx[..., 0]
        + dfy[..., 0]
    )

    r_hu = (
        q_t_interior[..., 1]
        + dfx[..., 1]
        + dfy[..., 1]
        + source_x
    )

    r_hv = (
        q_t_interior[..., 2]
        + dfx[..., 2]
        + dfy[..., 2]
        + source_y
    )

    return jnp.stack([r_h, r_hu, r_hv], axis=-1)


def swe_bathy_physics_loss(
    q: jnp.ndarray,
    q_t: jnp.ndarray,
    bathymetry: jnp.ndarray,
    *,
    dx: float = 0.15625,
    dy: float = 0.15625,
    g: float = 1.0,
) -> jnp.ndarray:
    residual = swe_bathy_residual(
        q,
        q_t,
        bathymetry,
        dx=dx,
        dy=dy,
        g=g,
    )
    return jnp.mean(residual**2)


def lake_at_rest_state(
    bathymetry: jnp.ndarray,
    *,
    eta0: float = 2.0,
) -> jnp.ndarray:
    if bathymetry.ndim == 4:
        b = bathymetry[..., 0]
    elif bathymetry.ndim == 3:
        b = bathymetry
    else:
        raise ValueError(
            "bathymetry must have shape (B,H,W) or (B,H,W,1)."
        )

    h = eta0 - b
    zeros = jnp.zeros_like(h)
    return jnp.stack([h, zeros, zeros], axis=-1)


__all__ = [
    "swe_bathy_fluxes",
    "well_balanced_bed_source",
    "swe_bathy_residual",
    "swe_bathy_physics_loss",
    "lake_at_rest_state",
]
