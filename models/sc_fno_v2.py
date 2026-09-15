"""
SC-FNO-v2.

This version is deliberately based on the EXACT successful FNO2d
architecture already used in this project.

The only architectural change is parameter sharing.

Original FNO2d:
    G1 -> G2 -> G3 -> G4

SC-FNO-v2:
    G_theta -> G_theta -> ... -> G_theta

where the same SpectralConv2d and the same 1x1 convolution are reused
at every composition step.

Everything else remains the same:
    - same input concatenation
    - same spatial grid
    - same bathymetry conditioning
    - same sinusoidal time embedding
    - same SpectralConv2d implementation
    - same GELU placement
    - same fc0
    - same fc1
    - same fc2

This gives us a clean experiment:

    ordinary FNO
        versus
    self-composing version of the same FNO.
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


class SCFNO2dV2(nn.Module):
    """
    Self-composing version of the project's original FNO2d.

    Parameters
    ----------
    num_channels:
        Number of output physical channels.

        For shallow water:
            [h_t, (hu)_t, (hv)_t]

        so num_channels = 3.

    modes1, modes2:
        Fourier modes. Same defaults as original FNO2d.

    width:
        Latent width. Same default as original FNO2d.

    compose_depth:
        Number of times the SAME Fourier/local block is applied.

    use_condition:
        If True, bathymetry is concatenated with the state and grid.

    use_time:
        If True, the same sinusoidal time embedding as the original
        FNO is added after fc0.
    """

    num_channels: int

    modes1: int = 12

    modes2: int = 12

    width: int = 64

    compose_depth: int = 4

    use_condition: bool = False

    use_time: bool = False

    # ============================================================
    # SETUP
    # ============================================================

    def setup(self):

        if self.compose_depth < 1:

            raise ValueError(
                "compose_depth must be >= 1."
            )

        # --------------------------------------------------------
        # EXACT SAME INPUT LIFTING AS ORIGINAL FNO2d
        # --------------------------------------------------------

        self.fc0 = nn.Dense(
            self.width
        )

        # --------------------------------------------------------
        # ONE SHARED SPECTRAL CONVOLUTION
        #
        # Original FNO:
        #     convs[0], convs[1], convs[2], convs[3]
        #
        # SC-FNO-v2:
        #     shared_conv reused K times
        # --------------------------------------------------------

        self.shared_conv = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )

        # --------------------------------------------------------
        # ONE SHARED LOCAL 1x1 CONVOLUTION
        #
        # Original:
        #     ws[0], ws[1], ws[2], ws[3]
        #
        # SC version:
        #     shared_w reused K times
        # --------------------------------------------------------

        self.shared_w = nn.Conv(
            self.width,
            kernel_size=(
                1,
                1,
            ),
        )

        # --------------------------------------------------------
        # EXACT SAME OUTPUT HEAD
        # --------------------------------------------------------

        self.fc1 = nn.Dense(
            128
        )

        self.fc2 = nn.Dense(
            self.num_channels
        )

    # ============================================================
    # FORWARD
    # ============================================================

    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[
            jnp.ndarray
        ] = None,
        c: Optional[
            jnp.ndarray
        ] = None,
        grid: Optional[
            jnp.ndarray
        ] = None,
    ):

        # --------------------------------------------------------
        # EXACT SAME INPUT HANDLING AS ORIGINAL FNO2d
        # --------------------------------------------------------

        if x.ndim == 3:

            x = x[
                ...,
                None
            ]

        N, H, W, _ = (
            x.shape
        )

        # ========================================================
        # SAME COORDINATE GRID
        # ========================================================

        if grid is None:

            gy = jnp.linspace(
                0.0,
                1.0,
                H,
            )

            gx = jnp.linspace(
                0.0,
                1.0,
                W,
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
                grid[
                    None,
                    ...
                ],
                (
                    N,
                    1,
                    1,
                    1,
                ),
            )

        # ========================================================
        # SAME INPUT CONCATENATION
        # ========================================================

        x = jnp.concatenate(
            [
                x,
                grid,
            ],
            axis=-1,
        )

        # ========================================================
        # SAME BATHYMETRY CONDITIONING
        # ========================================================

        if (
            self.use_condition
            and
            c is not None
        ):

            if c.ndim == 3:

                c = c[
                    ...,
                    None
                ]

            x = jnp.concatenate(
                [
                    x,
                    c,
                ],
                axis=-1,
            )

        # ========================================================
        # SAME INPUT LIFTING
        # ========================================================

        x = self.fc0(
            x
        )

        # ========================================================
        # SAME TIME CONDITIONING
        # ========================================================

        if (
            self.use_time
            and
            t is not None
        ):

            x = (
                x
                +
                _sinusoidal_time_embedding(
                    t,
                    self.width,
                )[
                    :,
                    None,
                    None,
                    :,
                ]
            )

        # ========================================================
        # SELF-COMPOSITION
        #
        # This mirrors the ORIGINAL FNO loop:
        #
        # for conv,w in first blocks:
        #     x = GELU(conv(x) + w(x))
        #
        # final block:
        #     x = conv(x) + w(x)
        #
        # Difference:
        #
        # the same conv and w are reused every time.
        # ========================================================

        for block_index in range(
            self.compose_depth
        ):

            x = (
                self.shared_conv(
                    x
                )
                +
                self.shared_w(
                    x
                )
            )

            # ----------------------------------------------------
            # Exactly like original FNO:
            # GELU after every block except the final block.
            # ----------------------------------------------------

            if (
                block_index
                <
                self.compose_depth
                -
                1
            ):

                x = jnn.gelu(
                    x
                )

        # ========================================================
        # EXACT SAME OUTPUT HEAD
        # ========================================================

        x = jnn.gelu(
            self.fc1(
                x
            )
        )

        x = self.fc2(
            x
        )

        return x


__all__ = [
    "SCFNO2dV2",
]