"""
Geometry-conditioned 2D U-FNO-style backbone.

Architecture:

    state/raw geometry
           |
        lifting
           +
    geometry encoder
           |
        latent z
           |
    standard FNO blocks
           |
    U-FNO blocks:
        spectral
        + pointwise
        + mini U-Net
           |
       projection

This is a 2D adaptation of the U-FNO concept for the present
continuous-time CFO vector-field setting.
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


# =====================================================================
# MINI U-NET
# =====================================================================

class MiniUNet2d(nn.Module):
    """
    Two-level U-Net branch used inside U-FNO blocks.

    Works naturally on 32x32 and 64x64 grids.
    """

    width: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        if (
            x.shape[1] % 4 != 0
            or
            x.shape[2] % 4 != 0
        ):

            raise ValueError(
                "MiniUNet2d requires both spatial "
                "dimensions to be divisible by 4."
            )

        # ---------------------------------------------------------
        # LEVEL 0
        # ---------------------------------------------------------

        x0 = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            padding="SAME",
            name="input_conv",
        )(
            x
        )

        x0 = jnn.gelu(
            x0
        )

        # ---------------------------------------------------------
        # DOWN 1
        # ---------------------------------------------------------

        d1 = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="down_1",
        )(
            x0
        )

        d1 = jnn.gelu(
            d1
        )

        d1 = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            padding="SAME",
            name="down_1_refine",
        )(
            d1
        )

        d1 = jnn.gelu(
            d1
        )

        # ---------------------------------------------------------
        # DOWN 2
        # ---------------------------------------------------------

        d2 = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="down_2",
        )(
            d1
        )

        d2 = jnn.gelu(
            d2
        )

        d2 = nn.Conv(
            self.width,
            kernel_size=(3, 3),
            padding="SAME",
            name="bottleneck",
        )(
            d2
        )

        d2 = jnn.gelu(
            d2
        )

        # ---------------------------------------------------------
        # UP 1
        # ---------------------------------------------------------

        u1 = nn.ConvTranspose(
            self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="up_1",
        )(
            d2
        )

        if (
            u1.shape[1:3]
            !=
            d1.shape[1:3]
        ):
            raise ValueError(
                "Unexpected U-Net shape mismatch "
                "at first skip connection."
            )

        u1 = jnn.gelu(
            u1
            +
            d1
        )

        # ---------------------------------------------------------
        # UP 2
        # ---------------------------------------------------------

        u2 = nn.ConvTranspose(
            self.width,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="SAME",
            name="up_2",
        )(
            u1
        )

        if (
            u2.shape[1:3]
            !=
            x0.shape[1:3]
        ):
            raise ValueError(
                "Unexpected U-Net shape mismatch "
                "at second skip connection."
            )

        u2 = jnn.gelu(
            u2
            +
            x0
        )

        return nn.Conv(
            self.width,
            kernel_size=(1, 1),
            padding="SAME",
            name="output_projection",
        )(
            u2
        )


# =====================================================================
# GEOMETRY U-FNO
# =====================================================================

class GeometryUFNO2d(nn.Module):
    """
    Geometry-conditioned U-FNO-style neural operator.

    The first blocks are standard FNO blocks.
    The last num_u_blocks also include a mini U-Net path.
    """

    num_channels: int

    modes1: int = 12
    modes2: int = 12

    width: int = 64
    num_blocks: int = 4

    num_u_blocks: int = 2

    geometry_width: int = 16
    geometry_depth: int = 2

    dx: float = 0.15625
    dy: float = 0.15625

    include_gradient_magnitude: bool = False

    use_time: bool = True

    def setup(
        self,
    ):

        if not (
            0
            <=
            self.num_u_blocks
            <=
            self.num_blocks
        ):

            raise ValueError(
                "num_u_blocks must satisfy "
                "0 <= num_u_blocks <= num_blocks."
            )

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

        self.unets = [
            MiniUNet2d(
                self.width
            )
            for _ in range(
                self.num_u_blocks
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
                "GeometryUFNO2d requires geometry c."
            )

        c = ensure_geometry_channels(
            c
        )

        batch_size, height, width, _ = (
            x.shape
        )

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

        z = (
            z
            +
            self.geometry_encoder(
                c
            )
        )

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

        u_start = (
            self.num_blocks
            -
            self.num_u_blocks
        )

        for block_index in range(
            self.num_blocks
        ):

            block_output = (
                self.convs[
                    block_index
                ](
                    z
                )
                +
                self.ws[
                    block_index
                ](
                    z
                )
            )

            if (
                block_index
                >=
                u_start
            ):

                unet_index = (
                    block_index
                    -
                    u_start
                )

                block_output = (
                    block_output
                    +
                    self.unets[
                        unet_index
                    ](
                        z
                    )
                )

            # Preserve the ordinary FNO convention:
            # activation after every block except the last.
            if (
                block_index
                <
                self.num_blocks
                -
                1
            ):

                z = jnn.gelu(
                    block_output
                )

            else:

                z = block_output

        z = jnn.gelu(
            self.fc1(
                z
            )
        )

        return self.fc2(
            z
        )


__all__ = [
    "MiniUNet2d",
    "GeometryUFNO2d",
]