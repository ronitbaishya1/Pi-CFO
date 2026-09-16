"""
Geometry-conditioned 2D FNO for Continuous Flow Operator training.

The ordinary FNO path is retained:

    [q, grid, raw geometry]
        -> lifting
        -> Fourier blocks

A learned geometry branch is added:

    geometry
        -> [g, g_x, g_y]
        -> local CNN encoder
        -> latent geometry features

The two lifted representations are added before the Fourier blocks.
"""

from __future__ import annotations

from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp
from flax import linen as nn

from models.fno import (
    SpectralConv2d,
    _sinusoidal_time_embedding,
)

from models.geometry_encoder import (
    GeometryEncoder2d,
    ensure_geometry_channels,
)


class GeometryFNO2d(nn.Module):
    """
    Geometry-conditioned FNO backbone.

    State:
        x = q

    Condition:
        c = raw geometry field

    For current SWE:
        c = bathymetry b(x,y)
    """

    num_channels: int

    modes1: int = 12
    modes2: int = 12

    width: int = 64
    num_blocks: int = 4

    geometry_width: int = 16
    geometry_depth: int = 2

    dx: float = 0.15625
    dy: float = 0.15625

    include_gradient_magnitude: bool = False

    use_time: bool = True

    def setup(
        self,
    ):

        self.geometry_encoder = (
            GeometryEncoder2d(
                output_width=self.width,
                hidden_width=self.geometry_width,
                depth=self.geometry_depth,
                dx=self.dx,
                dy=self.dy,
                include_gradient_magnitude=(
                    self.include_gradient_magnitude
                ),
            )
        )

        self.fc0 = nn.Dense(
            self.width
        )

        self.convs = [
            SpectralConv2d(
                self.width,
                self.width,
                self.modes1,
                self.modes2,
            )
            for _ in range(
                self.num_blocks
            )
        ]

        self.ws = [
            nn.Conv(
                self.width,
                kernel_size=(1, 1),
            )
            for _ in range(
                self.num_blocks
            )
        ]

        self.fc1 = nn.Dense(
            128
        )

        self.fc2 = nn.Dense(
            self.num_channels
        )

    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        c: Optional[jnp.ndarray] = None,
        grid: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        if x.ndim == 3:
            x = x[..., None]

        if c is None:
            raise ValueError(
                "GeometryFNO2d requires a geometry "
                "condition c."
            )

        c = ensure_geometry_channels(
            c
        )

        batch_size, height, width, _ = (
            x.shape
        )

        if (
            c.shape[0] != batch_size
            or
            c.shape[1] != height
            or
            c.shape[2] != width
        ):

            raise ValueError(
                "State and geometry spatial shapes "
                "must match."
            )

        # ---------------------------------------------------------
        # SAME GRID CONVENTION AS THE EXISTING FNO
        # ---------------------------------------------------------

        if grid is None:

            gy = jnp.linspace(
                0.0,
                1.0,
                height,
            )

            gx = jnp.linspace(
                0.0,
                1.0,
                width,
            )

            yy, xx = jnp.meshgrid(
                gy,
                gx,
                indexing="ij",
            )

            grid = jnp.stack(
                [
                    yy,
                    xx,
                ],
                axis=-1,
            )

            grid = jnp.tile(
                grid[None, ...],
                (
                    batch_size,
                    1,
                    1,
                    1,
                ),
            )

        # ---------------------------------------------------------
        # ORIGINAL RAW-CONDITION FNO PATH
        # ---------------------------------------------------------

        lifted_input = jnp.concatenate(
            [
                x,
                grid,
                c,
            ],
            axis=-1,
        )

        z = self.fc0(
            lifted_input
        )

        # ---------------------------------------------------------
        # GEOMETRY ENCODER PATH
        # ---------------------------------------------------------

        geometry_latent = (
            self.geometry_encoder(
                c
            )
        )

        z = (
            z
            +
            geometry_latent
        )

        # ---------------------------------------------------------
        # PHYSICAL TIME
        # ---------------------------------------------------------

        if (
            self.use_time
            and
            t is not None
        ):

            time_embedding = (
                _sinusoidal_time_embedding(
                    t,
                    self.width,
                )
            )

            z = (
                z
                +
                time_embedding[
                    :,
                    None,
                    None,
                    :,
                ]
            )

        # ---------------------------------------------------------
        # SAME FNO BLOCK STRUCTURE AS EXISTING BASELINE
        # ---------------------------------------------------------

        for conv, w in zip(
            self.convs[:-1],
            self.ws[:-1],
        ):

            z = jnn.gelu(
                conv(z)
                +
                w(z)
            )

        z = (
            self.convs[-1](z)
            +
            self.ws[-1](z)
        )

        z = jnn.gelu(
            self.fc1(
                z
            )
        )

        return self.fc2(
            z
        )


__all__ = [
    "GeometryFNO2d",
]