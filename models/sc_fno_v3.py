"""
SC-FNO-v3
=========

Self-composing FNO with residual iterative refinement.

Purpose
-------
Use the successful project's original FNO machinery, but make the
Fourier/local operator shared across composition steps and use each
composition as a SMALL REFINEMENT:

    z_{k+1} = z_k + alpha * Phi_theta(z_k)

where

    Phi_theta(z)
    =
    GELU(
        SpectralConv_theta(z)
        +
        W_theta(z)
    )

The same theta is reused for every composition step.

This is designed for Train-and-Unroll:

    K=1
      ->
    warm start K=2
      ->
    warm start K=3
      ->
    warm start K=4

The parameter tree is deliberately independent of compose_depth,
so checkpoints can be transferred directly between depths.

Input interface
---------------
Same interface as the project's original FNO:

    model(q, t, bathymetry)

State:
    q = [h, hu, hv]

Output:
    q_t = [h_t, (hu)_t, (hv)_t]

Important
---------
No well-balanced loss is implemented here.

This file changes ONLY the neural operator architecture.
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


# ================================================================
# SHARED RESIDUAL FNO UPDATE
# ================================================================

class SharedResidualFNOBlock(nn.Module):
    """
    One shared Fourier/local refinement block.

    Each application computes:

        correction =
            GELU(
                SpectralConv(x)
                +
                PointwiseConv(x)
            )

        x_new =
            x
            +
            alpha * correction

    The SAME SpectralConv and PointwiseConv parameters are reused
    at every self-composition step.
    """

    width: int = 64

    modes1: int = 12

    modes2: int = 12

    alpha: float = 0.25

    # ------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------

    def setup(self):

        self.spectral = SpectralConv2d(
            self.width,
            self.width,
            self.modes1,
            self.modes2,
        )

        self.pointwise = nn.Conv(
            self.width,
            kernel_size=(
                1,
                1,
            ),
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        spectral_update = self.spectral(
            x
        )

        local_update = self.pointwise(
            x
        )

        correction = (
            spectral_update
            +
            local_update
        )

        correction = jnn.gelu(
            correction
        )

        return (
            x
            +
            self.alpha
            *
            correction
        )


# ================================================================
# SC-FNO-v3
# ================================================================

class SCFNO2dV3(nn.Module):
    """
    Self-Composing FNO with residual iterative refinement.

    Architecture
    ------------

        q, grid, b
            |
            v
           fc0
            |
       + time embedding
            |
            v
           z0
            |
            v
        G_theta
            |
            v
           z1
            |
            v
        G_theta
            |
            v
           z2
            |
           ...
            |
            v
           zK
            |
           fc1
            |
          GELU
            |
           fc2
            |
            v
           q_t

    where

        z_{k+1}
        =
        z_k
        +
        alpha * Phi_theta(z_k)

    and the same theta is used for every k.
    """

    num_channels: int

    modes1: int = 12

    modes2: int = 12

    width: int = 64

    compose_depth: int = 1

    alpha: float = 0.25

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

        if self.alpha <= 0.0:

            raise ValueError(
                "alpha must be positive."
            )

        # --------------------------------------------------------
        # SAME INPUT LIFTING AS ORIGINAL FNO
        # --------------------------------------------------------

        self.fc0 = nn.Dense(
            self.width
        )

        # --------------------------------------------------------
        # ONE SHARED ITERATIVE UPDATE BLOCK
        # --------------------------------------------------------

        self.shared_operator = (
            SharedResidualFNOBlock(
                width=(
                    self.width
                ),
                modes1=(
                    self.modes1
                ),
                modes2=(
                    self.modes2
                ),
                alpha=(
                    self.alpha
                ),
            )
        )

        # --------------------------------------------------------
        # SAME OUTPUT PROJECTION STYLE AS ORIGINAL FNO
        # --------------------------------------------------------

        self.fc1 = nn.Dense(
            128
        )

        self.fc2 = nn.Dense(
            self.num_channels
        )

    # ============================================================
    # BUILD GRID
    # ============================================================

    @staticmethod
    def build_grid(
        batch_size: int,
        height: int,
        width: int,
        dtype,
    ) -> jnp.ndarray:

        gy = jnp.linspace(
            0.0,
            1.0,
            height,
            dtype=dtype,
        )

        gx = jnp.linspace(
            0.0,
            1.0,
            width,
            dtype=dtype,
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

        grid = jnp.broadcast_to(
            grid[
                None,
                ...
            ],
            (
                batch_size,
                height,
                width,
                2,
            ),
        )

        return grid

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
    ) -> jnp.ndarray:

        # --------------------------------------------------------
        # STATE CHANNEL
        # --------------------------------------------------------

        if x.ndim == 3:

            x = x[
                ...,
                None
            ]

        batch_size = x.shape[
            0
        ]

        height = x.shape[
            1
        ]

        spatial_width = x.shape[
            2
        ]

        dtype = x.dtype

        # ========================================================
        # GRID
        # ========================================================

        if grid is None:

            grid = self.build_grid(
                batch_size,
                height,
                spatial_width,
                dtype,
            )

        # ========================================================
        # INPUT FEATURES
        # ========================================================

        features = [
            x,
            grid,
        ]

        # ========================================================
        # BATHYMETRY CONDITION
        # ========================================================

        if self.use_condition:

            if c is None:

                raise ValueError(
                    "SCFNO2dV3 was created with "
                    "use_condition=True, but no "
                    "bathymetry condition was supplied."
                )

            condition = jnp.asarray(
                c,
                dtype=dtype,
            )

            if condition.ndim == 3:

                condition = condition[
                    ...,
                    None
                ]

            if (
                condition.shape[
                    0
                ]
                !=
                batch_size
            ):

                raise ValueError(
                    "Bathymetry batch size "
                    "does not match state."
                )

            if (
                condition.shape[
                    1
                ]
                !=
                height
                or
                condition.shape[
                    2
                ]
                !=
                spatial_width
            ):

                raise ValueError(
                    "Bathymetry spatial shape "
                    "does not match state."
                )

            features.append(
                condition
            )

        # ========================================================
        # CONCATENATE STATE + GRID + BATHYMETRY
        # ========================================================

        network_input = jnp.concatenate(
            features,
            axis=-1,
        )

        # ========================================================
        # LIFT
        # ========================================================

        latent = self.fc0(
            network_input
        )

        # ========================================================
        # CONTINUOUS TIME CONDITIONING
        # ========================================================

        if self.use_time:

            if t is None:

                raise ValueError(
                    "SCFNO2dV3 was created with "
                    "use_time=True, but no time "
                    "value was supplied."
                )

            latent = (
                latent
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
        # SAME block reused compose_depth times.
        # ========================================================

        for _ in range(
            self.compose_depth
        ):

            latent = (
                self.shared_operator(
                    latent
                )
            )

        # ========================================================
        # OUTPUT PROJECTION
        # ========================================================

        latent = jnn.gelu(
            self.fc1(
                latent
            )
        )

        output = self.fc2(
            latent
        )

        return output


__all__ = [
    "SharedResidualFNOBlock",
    "SCFNO2dV3",
]