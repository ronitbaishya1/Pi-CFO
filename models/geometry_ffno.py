"""
Geometry-conditioned Factorized Fourier Neural Operator.

Main F-FNO ideas used here:

1. Fourier transforms are factorized by spatial dimension.
2. The factorized spectral kernel is shared across operator blocks.
3. Each block uses a residual feed-forward update.
4. Geometry encoding is kept identical to GeometryFNO2d and
   GeometryUFNO2d.

This keeps the architecture comparison controlled.
"""

from __future__ import annotations

from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp
from flax import linen as nn

from models.fno import (
    _sinusoidal_time_embedding,
)

from models.geometry_encoder import (
    GeometryEncoder2d,
    ensure_geometry_channels,
)


# =====================================================================
# FACTORIZED SPECTRAL OPERATOR
# =====================================================================

class FactorizedSpectralConv2d(nn.Module):
    """
    Separable spectral operator:

        K(z)
        =
        IFFT_x(R_x FFT_x(z))
        +
        IFFT_y(R_y FFT_y(z))

    Input/output shape:

        (B,H,W,C)
    """

    width: int

    modes1: int = 12
    modes2: int = 12

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        scale = (
            1.0
            /
            (
                self.width
                *
                self.width
            )
        )

        # ---------------------------------------------------------
        # X-DIRECTION SPECTRAL WEIGHTS
        # ---------------------------------------------------------

        wx_real = self.param(
            "wx_real",
            nn.initializers.normal(
                stddev=scale
            ),
            (
                self.modes1,
                self.width,
                self.width,
            ),
        )

        wx_imag = self.param(
            "wx_imag",
            nn.initializers.normal(
                stddev=scale
            ),
            (
                self.modes1,
                self.width,
                self.width,
            ),
        )

        wx = (
            wx_real
            +
            1j
            *
            wx_imag
        )

        # ---------------------------------------------------------
        # Y-DIRECTION SPECTRAL WEIGHTS
        # ---------------------------------------------------------

        wy_real = self.param(
            "wy_real",
            nn.initializers.normal(
                stddev=scale
            ),
            (
                self.modes2,
                self.width,
                self.width,
            ),
        )

        wy_imag = self.param(
            "wy_imag",
            nn.initializers.normal(
                stddev=scale
            ),
            (
                self.modes2,
                self.width,
                self.width,
            ),
        )

        wy = (
            wy_real
            +
            1j
            *
            wy_imag
        )

        batch_size, height, width, channels = (
            x.shape
        )

        # ---------------------------------------------------------
        # X FACTORIZATION
        #
        # axis 1 follows the x-direction convention used by the
        # current SWE physics implementation.
        # ---------------------------------------------------------

        x_ft = jnp.fft.rfft(
            x,
            axis=1,
        )

        x_out_ft = jnp.zeros(
            (
                batch_size,
                x_ft.shape[1],
                width,
                channels,
            ),
            dtype=x_ft.dtype,
        )

        mx = min(
            self.modes1,
            x_ft.shape[1],
        )

        x_modes = jnp.einsum(
            "bmwc,mco->bmwo",
            x_ft[
                :,
                :mx,
                :,
                :,
            ],
            wx[
                :mx,
                :,
                :,
            ],
        )

        x_out_ft = (
            x_out_ft
            .at[
                :,
                :mx,
                :,
                :,
            ]
            .set(
                x_modes
            )
        )

        x_branch = jnp.fft.irfft(
            x_out_ft,
            n=height,
            axis=1,
        )

        # ---------------------------------------------------------
        # Y FACTORIZATION
        # ---------------------------------------------------------

        y_ft = jnp.fft.rfft(
            x,
            axis=2,
        )

        y_out_ft = jnp.zeros(
            (
                batch_size,
                height,
                y_ft.shape[2],
                channels,
            ),
            dtype=y_ft.dtype,
        )

        my = min(
            self.modes2,
            y_ft.shape[2],
        )

        y_modes = jnp.einsum(
            "bhmc,mco->bhmo",
            y_ft[
                :,
                :,
                :my,
                :,
            ],
            wy[
                :my,
                :,
                :,
            ],
        )

        y_out_ft = (
            y_out_ft
            .at[
                :,
                :,
                :my,
                :,
            ]
            .set(
                y_modes
            )
        )

        y_branch = jnp.fft.irfft(
            y_out_ft,
            n=width,
            axis=2,
        )

        return (
            x_branch
            +
            y_branch
        )


# =====================================================================
# GEOMETRY F-FNO
# =====================================================================

class GeometryFFNO2d(nn.Module):
    """
    Geometry-conditioned F-FNO backbone.
    """

    num_channels: int

    modes1: int = 12
    modes2: int = 12

    width: int = 64
    num_blocks: int = 4

    expansion: int = 2

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

        # Shared spectral kernel, as motivated by F-FNO.
        self.factorized_spectral = (
            FactorizedSpectralConv2d(
                width=self.width,
                modes1=self.modes1,
                modes2=self.modes2,
            )
        )

        self.feedforward_1 = [
            nn.Dense(
                self.width
                *
                self.expansion
            )
            for _ in range(
                self.num_blocks
            )
        ]

        self.feedforward_2 = [
            nn.Dense(
                self.width
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
                "GeometryFFNO2d requires geometry c."
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

        # ---------------------------------------------------------
        # FACTORIZED RESIDUAL OPERATOR BLOCKS
        # ---------------------------------------------------------

        for block_index in range(
            self.num_blocks
        ):

            spectral = (
                self.factorized_spectral(
                    z
                )
            )

            update = (
                self.feedforward_1[
                    block_index
                ](
                    spectral
                )
            )

            update = jnn.gelu(
                update
            )

            update = (
                self.feedforward_2[
                    block_index
                ](
                    update
                )
            )

            update = jnn.gelu(
                update
            )

            z = (
                z
                +
                update
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
    "FactorizedSpectralConv2d",
    "GeometryFFNO2d",
]