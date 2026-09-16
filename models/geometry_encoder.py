"""
Generic geometry feature construction and encoding for CFO neural operators.

For a scalar geometry field g(x,y), the default features are

    [g, g_x, g_y]

For the current shallow-water problem:

    g = b(x,y)

Later the same interface can be used for:

    g = phi(x,y,t)      moving interface / level set
    g = SDF(x,y)        solid obstacle geometry

The encoder converts local geometry information into a latent feature
field that can be fused with an FNO, U-FNO or F-FNO backbone.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp
from flax import linen as nn


# =====================================================================
# SHAPE HANDLING
# =====================================================================

def ensure_geometry_channels(
    geometry: jnp.ndarray,
) -> jnp.ndarray:
    """
    Ensure geometry has shape:

        (B, H, W, C)
    """

    if geometry.ndim == 3:
        geometry = geometry[..., None]

    if geometry.ndim != 4:
        raise ValueError(
            "geometry must have shape "
            "(B,H,W) or (B,H,W,C)."
        )

    return geometry


# =====================================================================
# FINITE-DIFFERENCE GEOMETRY GRADIENTS
# =====================================================================

def geometry_ddx(
    geometry: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """
    First derivative along spatial axis 1.

    This matches the x-direction convention used by
    utils/physics_swe_bathy.py.
    """

    geometry = ensure_geometry_channels(
        geometry
    )

    if geometry.shape[1] < 2:
        raise ValueError(
            "Need at least two grid points in x."
        )

    left = (
        geometry[:, 1:2, :, :]
        -
        geometry[:, 0:1, :, :]
    ) / dx

    center = (
        geometry[:, 2:, :, :]
        -
        geometry[:, :-2, :, :]
    ) / (2.0 * dx)

    right = (
        geometry[:, -1:, :, :]
        -
        geometry[:, -2:-1, :, :]
    ) / dx

    return jnp.concatenate(
        [
            left,
            center,
            right,
        ],
        axis=1,
    )


def geometry_ddy(
    geometry: jnp.ndarray,
    dy: float,
) -> jnp.ndarray:
    """
    First derivative along spatial axis 2.

    This matches the y-direction convention used by
    utils/physics_swe_bathy.py.
    """

    geometry = ensure_geometry_channels(
        geometry
    )

    if geometry.shape[2] < 2:
        raise ValueError(
            "Need at least two grid points in y."
        )

    bottom = (
        geometry[:, :, 1:2, :]
        -
        geometry[:, :, 0:1, :]
    ) / dy

    center = (
        geometry[:, :, 2:, :]
        -
        geometry[:, :, :-2, :]
    ) / (2.0 * dy)

    top = (
        geometry[:, :, -1:, :]
        -
        geometry[:, :, -2:-1, :]
    ) / dy

    return jnp.concatenate(
        [
            bottom,
            center,
            top,
        ],
        axis=2,
    )


# =====================================================================
# GEOMETRY FEATURE MAP
# =====================================================================

def build_geometry_features(
    geometry: jnp.ndarray,
    *,
    dx: float,
    dy: float,
    include_gradient_magnitude: bool = False,
) -> jnp.ndarray:
    """
    Construct geometry features.

    Default:

        [g, g_x, g_y]

    Optional:

        [g, g_x, g_y, |grad g|]
    """

    geometry = ensure_geometry_channels(
        geometry
    )

    geometry_x = geometry_ddx(
        geometry,
        dx,
    )

    geometry_y = geometry_ddy(
        geometry,
        dy,
    )

    features = [
        geometry,
        geometry_x,
        geometry_y,
    ]

    if include_gradient_magnitude:

        gradient_magnitude = jnp.sqrt(
            geometry_x**2
            +
            geometry_y**2
            +
            1.0e-12
        )

        features.append(
            gradient_magnitude
        )

    return jnp.concatenate(
        features,
        axis=-1,
    )


# =====================================================================
# LEARNED GEOMETRY ENCODER
# =====================================================================

class GeometryEncoder2d(nn.Module):
    """
    Small local CNN that converts

        [g, g_x, g_y]

    into a latent geometry representation.

    The output width is normally chosen to equal the operator width,
    allowing the geometry representation to be added directly to the
    lifted physical state.
    """

    output_width: int
    hidden_width: int = 16
    depth: int = 2

    dx: float = 0.15625
    dy: float = 0.15625

    include_gradient_magnitude: bool = False

    @nn.compact
    def __call__(
        self,
        geometry: jnp.ndarray,
    ) -> jnp.ndarray:

        features = build_geometry_features(
            geometry,
            dx=self.dx,
            dy=self.dy,
            include_gradient_magnitude=(
                self.include_gradient_magnitude
            ),
        )

        z = features

        for layer_index in range(
            self.depth
        ):

            z = nn.Conv(
                features=self.hidden_width,
                kernel_size=(3, 3),
                padding="SAME",
                name=f"geometry_conv_{layer_index}",
            )(
                z
            )

            z = jnn.gelu(
                z
            )

        z = nn.Conv(
            features=self.output_width,
            kernel_size=(1, 1),
            padding="SAME",
            name="geometry_projection",
        )(
            z
        )

        return z


__all__ = [
    "ensure_geometry_channels",
    "geometry_ddx",
    "geometry_ddy",
    "build_geometry_features",
    "GeometryEncoder2d",
]