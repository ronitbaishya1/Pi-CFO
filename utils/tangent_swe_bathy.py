"""
Utilities for directional tangent analysis of the variable-bottom
shallow-water equations.

The existing well-balanced SWE residual is

    q_t + div(F(q)) + S(q,b) = 0

so the corresponding discrete physical vector field is

    q_t = f_SWE(q,b)
        = -div(F(q)) - S(q,b).

The functions in this file allow JAX to calculate

    J_SWE(q,b) v

without ever constructing the full Jacobian.

Two perturbation families are provided:

1. Random perturbation

       v_random

   Spatially uncorrelated Gaussian noise.

2. Smooth perturbation

       v_smooth

   The SAME Gaussian noise field after channel-wise Gaussian
   spatial filtering.

Both directions are zeroed at the domain boundary and normalized
independently for every batch member.
"""

from __future__ import annotations

import jax

import jax.numpy as jnp

from utils.physics_swe_bathy import (
    swe_bathy_fluxes,
    ddx_center,
    ddy_center,
    well_balanced_bed_source,
)


# =====================================================================
# DISCRETE PHYSICAL SWE VECTOR FIELD
# =====================================================================

def swe_bathy_rhs_interior(
    q: jnp.ndarray,
    bathymetry: jnp.ndarray,
    *,
    dx: float = 0.15625,
    dy: float = 0.15625,
    g: float = 1.0,
) -> jnp.ndarray:
    """
    Return the discrete variable-bottom SWE vector field on the
    common interior grid.

    Existing residual:

        R =
        q_t
        +
        div(F)
        +
        bed_source

    Therefore:

        q_t =
        -div(F)
        -bed_source.

    Parameters
    ----------
    q:
        Shape (B,H,W,3)

    bathymetry:
        Shape (B,H,W,1) or (B,H,W)

    Returns
    -------
    rhs:
        Shape (B,H-2,W-2,3)
    """

    # -----------------------------------------------------------------
    # PHYSICAL FLUXES
    # -----------------------------------------------------------------

    fx, fy = swe_bathy_fluxes(
        q,
        g=g,
    )

    # -----------------------------------------------------------------
    # CENTERED SPATIAL DERIVATIVES
    # -----------------------------------------------------------------

    dfx = ddx_center(
        fx,
        dx,
    )

    dfy = ddy_center(
        fy,
        dy,
    )

    # -----------------------------------------------------------------
    # WELL-BALANCED BED SOURCE
    # -----------------------------------------------------------------

    (
        source_x,
        source_y,
    ) = well_balanced_bed_source(
        q,
        bathymetry,
        dx=dx,
        dy=dy,
        g=g,
    )

    # -----------------------------------------------------------------
    # CONTINUITY
    # -----------------------------------------------------------------

    rhs_h = -(
        dfx[..., 0]
        +
        dfy[..., 0]
    )

    # -----------------------------------------------------------------
    # X MOMENTUM
    # -----------------------------------------------------------------

    rhs_hu = -(
        dfx[..., 1]
        +
        dfy[..., 1]
        +
        source_x
    )

    # -----------------------------------------------------------------
    # Y MOMENTUM
    # -----------------------------------------------------------------

    rhs_hv = -(
        dfx[..., 2]
        +
        dfy[..., 2]
        +
        source_y
    )

    return jnp.stack(
        [
            rhs_h,
            rhs_hu,
            rhs_hv,
        ],
        axis=-1,
    )


# =====================================================================
# BOUNDARY HANDLING
# =====================================================================

def zero_tangent_boundary(
    direction: jnp.ndarray,
) -> jnp.ndarray:
    """
    Set tangent perturbations to zero on all four boundaries.

    Input:
        (B,H,W,C)
    """

    if direction.ndim != 4:

        raise ValueError(
            "direction must have shape "
            "(B,H,W,C)."
        )

    direction = direction.at[
        :,
        0,
        :,
        :,
    ].set(
        0.0
    )

    direction = direction.at[
        :,
        -1,
        :,
        :,
    ].set(
        0.0
    )

    direction = direction.at[
        :,
        :,
        0,
        :,
    ].set(
        0.0
    )

    direction = direction.at[
        :,
        :,
        -1,
        :,
    ].set(
        0.0
    )

    return direction


# =====================================================================
# NORMALIZATION
# =====================================================================

def normalize_tangent_direction(
    direction: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """
    Normalize every batch member independently.

    The complete state perturbation satisfies approximately

        ||v||_2 = 1.
    """

    norm = jnp.sqrt(
        jnp.sum(
            direction**2,
            axis=(
                1,
                2,
                3,
            ),
            keepdims=True,
        )
        +
        eps
    )

    return (
        direction
        /
        norm
    )


# =====================================================================
# GENERIC PREPARATION
# =====================================================================

def prepare_tangent_direction(
    direction: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """
    Boundary-zero and normalize an arbitrary tangent direction.

    Retained as the generic helper used by tangent-consistent
    training.
    """

    direction = zero_tangent_boundary(
        direction
    )

    direction = normalize_tangent_direction(
        direction,
        eps=eps,
    )

    return direction


# =====================================================================
# GAUSSIAN KERNEL
# =====================================================================

def gaussian_kernel_2d(
    *,
    sigma: float,
    kernel_size: int,
    dtype=jnp.float32,
) -> jnp.ndarray:
    """
    Construct a normalized 2D Gaussian kernel.

    For the initial 32x32 SWE experiments we use

        sigma = 1.5 cells
        kernel_size = 7.
    """

    if sigma <= 0.0:

        raise ValueError(
            "sigma must be > 0."
        )

    if kernel_size <= 0:

        raise ValueError(
            "kernel_size must be positive."
        )

    if kernel_size % 2 == 0:

        raise ValueError(
            "kernel_size must be odd."
        )

    radius = (
        kernel_size
        //
        2
    )

    coordinates = jnp.arange(
        -radius,
        radius + 1,
        dtype=dtype,
    )

    xx, yy = jnp.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    kernel = jnp.exp(
        -(
            xx**2
            +
            yy**2
        )
        /
        (
            2.0
            *
            sigma**2
        )
    )

    kernel = (
        kernel
        /
        jnp.sum(
            kernel
        )
    )

    return kernel


# =====================================================================
# CHANNEL-WISE GAUSSIAN SMOOTHING
# =====================================================================

def gaussian_smooth_channels(
    field: jnp.ndarray,
    *,
    sigma: float = 1.5,
    kernel_size: int = 7,
) -> jnp.ndarray:
    """
    Smooth every physical channel independently.

    Input
    -----
        (B,H,W,C)

    The convolution is depth-wise:

        h perturbation  -> smoothed independently
        hu perturbation -> smoothed independently
        hv perturbation -> smoothed independently

    No mixing occurs between state channels.
    """

    if field.ndim != 4:

        raise ValueError(
            "field must have shape "
            "(B,H,W,C)."
        )

    num_channels = int(
        field.shape[
            -1
        ]
    )

    kernel = gaussian_kernel_2d(
        sigma=sigma,
        kernel_size=kernel_size,
        dtype=field.dtype,
    )

    # -------------------------------------------------------------
    # JAX convolution kernel format for NHWC/HWIO.
    #
    # feature_group_count=C gives a depth-wise convolution.
    #
    # Kernel shape:
    #
    #     (KH,KW,1,C)
    # -------------------------------------------------------------

    kernel = kernel[
        :,
        :,
        None,
        None,
    ]

    kernel = jnp.tile(
        kernel,
        (
            1,
            1,
            1,
            num_channels,
        ),
    )

    smoothed = jax.lax.conv_general_dilated(
        lhs=field,
        rhs=kernel,
        window_strides=(
            1,
            1,
        ),
        padding="SAME",
        dimension_numbers=(
            "NHWC",
            "HWIO",
            "NHWC",
        ),
        feature_group_count=(
            num_channels
        ),
    )

    return smoothed


# =====================================================================
# RANDOM PERTURBATION
# =====================================================================

def random_tangent_direction(
    raw_noise: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """
    Construct the normalized random perturbation

        v_random = xi / ||xi||

    with zero boundary perturbation.
    """

    return prepare_tangent_direction(
        raw_noise,
        eps=eps,
    )


# =====================================================================
# SMOOTH PERTURBATION
# =====================================================================

def smooth_tangent_direction(
    raw_noise: jnp.ndarray,
    *,
    sigma: float = 1.5,
    kernel_size: int = 7,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """
    Construct

        v_smooth =
        GaussianBlur(xi)
        /
        ||GaussianBlur(xi)||.

    IMPORTANT:

    This uses the SAME raw Gaussian field xi as the corresponding
    random perturbation.
    """

    smoothed = gaussian_smooth_channels(
        raw_noise,
        sigma=sigma,
        kernel_size=kernel_size,
    )

    return prepare_tangent_direction(
        smoothed,
        eps=eps,
    )


# =====================================================================
# RANDOM + SMOOTH PAIR
# =====================================================================

def random_and_smooth_directions(
    raw_noise: jnp.ndarray,
    *,
    sigma: float = 1.5,
    kernel_size: int = 7,
    eps: float = 1.0e-12,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
]:
    """
    Return a paired perturbation experiment:

        random direction
        smooth direction

    created from exactly the same Gaussian noise field.
    """

    random_direction = (
        random_tangent_direction(
            raw_noise,
            eps=eps,
        )
    )

    smooth_direction = (
        smooth_tangent_direction(
            raw_noise,
            sigma=sigma,
            kernel_size=kernel_size,
            eps=eps,
        )
    )

    return (
        random_direction,
        smooth_direction,
    )


# =====================================================================
# TRAINING LOSS METRIC
# =====================================================================

def symmetric_normalized_tangent_error(
    model_jvp: jnp.ndarray,
    physics_jvp: jnp.ndarray,
    *,
    eps: float = 1.0e-12,
) -> jnp.ndarray:
    """
    Symmetric normalized tangent discrepancy.

    For each batch member:

                       ||a-b||^2
        E = --------------------------------
            ||a||^2 + ||b||^2 + eps

    where

        a = J_model v
        b = J_SWE v.

    Interpretation
    --------------

        0:
            perfect agreement

        approximately 1:
            strong disagreement / one response much smaller

        approximately 2:
            opposite responses of similar magnitude
    """

    difference_squared = jnp.sum(
        (
            model_jvp
            -
            physics_jvp
        )
        ** 2,
        axis=(
            1,
            2,
            3,
        ),
    )

    model_squared = jnp.sum(
        model_jvp**2,
        axis=(
            1,
            2,
            3,
        ),
    )

    physics_squared = jnp.sum(
        physics_jvp**2,
        axis=(
            1,
            2,
            3,
        ),
    )

    score = (
        difference_squared
        /
        (
            model_squared
            +
            physics_squared
            +
            eps
        )
    )

    return jnp.mean(
        score
    )


__all__ = [
    "swe_bathy_rhs_interior",
    "zero_tangent_boundary",
    "normalize_tangent_direction",
    "prepare_tangent_direction",
    "gaussian_kernel_2d",
    "gaussian_smooth_channels",
    "random_tangent_direction",
    "smooth_tangent_direction",
    "random_and_smooth_directions",
    "symmetric_normalized_tangent_error",
]