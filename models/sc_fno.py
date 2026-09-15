"""
Self-Composing Fourier Neural Operator for 2D PDE dynamics.

Purpose
-------
This model is a drop-in neural-vector-field backbone for CFO.

Input
-----
q:
    (batch, nx, ny, num_channels)

For the shallow-water problem:

    q = [h, hu, hv]

Optional condition:
    b(x,y)

Optional continuous time:
    t

Output
------
q_t:

    [h_t, (hu)_t, (hv)_t]

Self-composition
----------------
A single shared Fourier operator block G_theta is applied repeatedly:

    z_1 = G_theta(z_0)
    z_2 = G_theta(z_1)
    ...
    z_K = G_theta(z_{K-1})

The SAME parameters are reused at every composition step.

This is intentionally separate from models/fno.py so the original
FNO baseline remains frozen and reproducible.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from flax import linen as nn


# ================================================================
# SPECTRAL CONVOLUTION
# ================================================================

class SpectralConv2d(nn.Module):
    """
    2D Fourier convolution using low-frequency spectral modes.

    Input/output convention:

        (batch, nx, ny, channels)
    """

    in_channels: int

    out_channels: int

    modes1: int = 12

    modes2: int = 12

    # ------------------------------------------------------------
    # COMPLEX MULTIPLICATION
    # ------------------------------------------------------------

    @staticmethod
    def complex_multiply(
        x,
        weights,
    ):
        """
        x:
            (batch, mx, my, in_channels)

        weights:
            (mx, my, in_channels, out_channels)

        output:
            (batch, mx, my, out_channels)
        """

        return jnp.einsum(
            "bxyi,xyio->bxyo",
            x,
            weights,
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    @nn.compact
    def __call__(
        self,
        x,
    ):

        batch_size = x.shape[
            0
        ]

        nx = x.shape[
            1
        ]

        ny = x.shape[
            2
        ]

        # ========================================================
        # FFT
        # ========================================================

        x_ft = jnp.fft.rfft2(
            x,
            axes=(
                1,
                2,
            ),
        )

        ny_fourier = x_ft.shape[
            2
        ]

        # ========================================================
        # SAFE NUMBER OF MODES
        # ========================================================

        modes1 = min(
            self.modes1,
            nx // 2,
        )

        modes2 = min(
            self.modes2,
            ny_fourier,
        )

        # ========================================================
        # WEIGHT INITIALIZATION SCALE
        # ========================================================

        scale = (
            1.0
            /
            max(
                self.in_channels
                *
                self.out_channels,
                1,
            )
        )

        initializer = nn.initializers.normal(
            stddev=scale
        )

        # ========================================================
        # POSITIVE X-FREQUENCY WEIGHTS
        # ========================================================

        weight_pos_real = self.param(
            "weight_pos_real",
            initializer,
            (
                modes1,
                modes2,
                self.in_channels,
                self.out_channels,
            ),
        )

        weight_pos_imag = self.param(
            "weight_pos_imag",
            initializer,
            (
                modes1,
                modes2,
                self.in_channels,
                self.out_channels,
            ),
        )

        weight_pos = (
            weight_pos_real
            +
            1j
            *
            weight_pos_imag
        )

        # ========================================================
        # NEGATIVE X-FREQUENCY WEIGHTS
        # ========================================================

        weight_neg_real = self.param(
            "weight_neg_real",
            initializer,
            (
                modes1,
                modes2,
                self.in_channels,
                self.out_channels,
            ),
        )

        weight_neg_imag = self.param(
            "weight_neg_imag",
            initializer,
            (
                modes1,
                modes2,
                self.in_channels,
                self.out_channels,
            ),
        )

        weight_neg = (
            weight_neg_real
            +
            1j
            *
            weight_neg_imag
        )

        # ========================================================
        # EMPTY FOURIER OUTPUT
        # ========================================================

        output_ft = jnp.zeros(
            (
                batch_size,
                nx,
                ny_fourier,
                self.out_channels,
            ),
            dtype=x_ft.dtype,
        )

        # ========================================================
        # POSITIVE MODES
        # ========================================================

        positive_output = (
            self.complex_multiply(
                x_ft[
                    :,
                    :modes1,
                    :modes2,
                    :,
                ],
                weight_pos,
            )
        )

        output_ft = output_ft.at[
            :,
            :modes1,
            :modes2,
            :,
        ].set(
            positive_output
        )

        # ========================================================
        # NEGATIVE MODES
        # ========================================================

        negative_output = (
            self.complex_multiply(
                x_ft[
                    :,
                    -modes1:,
                    :modes2,
                    :,
                ],
                weight_neg,
            )
        )

        output_ft = output_ft.at[
            :,
            -modes1:,
            :modes2,
            :,
        ].set(
            negative_output
        )

        # ========================================================
        # INVERSE FFT
        # ========================================================

        output = jnp.fft.irfft2(
            output_ft,
            s=(
                nx,
                ny,
            ),
            axes=(
                1,
                2,
            ),
        )

        return output


# ================================================================
# SHARED SELF-COMPOSING BLOCK
# ================================================================

class SharedFNOBlock(nn.Module):
    """
    One shared Fourier block.

    The exact same parameters are reused every time this block
    is applied during self-composition.
    """

    width: int = 64

    modes1: int = 12

    modes2: int = 12

    residual_scale: float = 0.5

    @nn.compact
    def __call__(
        self,
        x,
    ):

        # ========================================================
        # NORMALIZE LATENT FEATURES
        # ========================================================

        normalized = nn.LayerNorm(
            name="layer_norm"
        )(
            x
        )

        # ========================================================
        # GLOBAL FOURIER UPDATE
        # ========================================================

        spectral_update = (
            SpectralConv2d(
                in_channels=(
                    self.width
                ),
                out_channels=(
                    self.width
                ),
                modes1=(
                    self.modes1
                ),
                modes2=(
                    self.modes2
                ),
                name="spectral",
            )(
                normalized
            )
        )

        # ========================================================
        # LOCAL 1x1 UPDATE
        # ========================================================

        local_update = nn.Conv(
            features=(
                self.width
            ),
            kernel_size=(
                1,
                1,
            ),
            padding="SAME",
            name="pointwise",
        )(
            normalized
        )

        # ========================================================
        # COMBINE GLOBAL + LOCAL INFORMATION
        # ========================================================

        update = (
            spectral_update
            +
            local_update
        )

        update = nn.gelu(
            update
        )

        # ========================================================
        # SECOND LOCAL MIXING
        # ========================================================

        update = nn.Conv(
            features=(
                self.width
            ),
            kernel_size=(
                1,
                1,
            ),
            padding="SAME",
            name="mix",
        )(
            update
        )

        # ========================================================
        # RESIDUAL REFINEMENT
        # ========================================================

        return (
            x
            +
            self.residual_scale
            *
            update
        )


# ================================================================
# SELF-COMPOSING FNO
# ================================================================

class SCFNO2d(nn.Module):
    """
    Self-Composing Fourier Neural Operator.

    Interface intentionally matches the existing FNO usage:

        model(q, t, condition)

    so CFO does not need to know whether its backbone is FNO
    or SC-FNO.
    """

    num_channels: int

    modes1: int = 12

    modes2: int = 12

    width: int = 64

    compose_depth: int = 4

    residual_scale: float = 0.5

    use_condition: bool = False

    use_time: bool = False

    # ------------------------------------------------------------
    # COORDINATE GRID
    # ------------------------------------------------------------

    @staticmethod
    def make_grid(
        batch_size,
        nx,
        ny,
        dtype,
    ):

        x_coordinates = jnp.linspace(
            0.0,
            1.0,
            nx,
            dtype=dtype,
        )

        y_coordinates = jnp.linspace(
            0.0,
            1.0,
            ny,
            dtype=dtype,
        )

        X, Y = jnp.meshgrid(
            x_coordinates,
            y_coordinates,
            indexing="ij",
        )

        grid = jnp.stack(
            [
                X,
                Y,
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
                nx,
                ny,
                2,
            ),
        )

        return grid

    # ------------------------------------------------------------
    # TIME FIELD
    # ------------------------------------------------------------

    @staticmethod
    def make_time_field(
        t,
        batch_size,
        nx,
        ny,
        dtype,
    ):

        time = jnp.asarray(
            t,
            dtype=dtype,
        )

        # Scalar time
        if time.ndim == 0:

            time = jnp.full(
                (
                    batch_size,
                ),
                time,
                dtype=dtype,
            )

        # (B,1) -> (B,)
        elif (
            time.ndim == 2
            and
            time.shape[
                -1
            ]
            ==
            1
        ):

            time = time[
                :,
                0
            ]

        # Ensure shape is (B,)
        time = jnp.reshape(
            time,
            (
                batch_size,
            ),
        )

        time = time[
            :,
            None,
            None,
            None,
        ]

        return jnp.broadcast_to(
            time,
            (
                batch_size,
                nx,
                ny,
                1,
            ),
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    @nn.compact
    def __call__(
        self,
        x,
        t: Optional[
            jnp.ndarray
        ] = None,
        c: Optional[
            jnp.ndarray
        ] = None,
    ):

        if (
            self.compose_depth
            <
            1
        ):

            raise ValueError(
                "compose_depth must "
                "be >= 1."
            )

        batch_size = x.shape[
            0
        ]

        nx = x.shape[
            1
        ]

        ny = x.shape[
            2
        ]

        dtype = x.dtype

        # ========================================================
        # INPUT FEATURES
        # ========================================================

        features = [
            x
        ]

        # --------------------------------------------------------
        # Spatial coordinates
        # --------------------------------------------------------

        grid = self.make_grid(
            batch_size,
            nx,
            ny,
            dtype,
        )

        features.append(
            grid
        )

        # --------------------------------------------------------
        # Bathymetry condition
        # --------------------------------------------------------

        if self.use_condition:

            if c is None:

                raise ValueError(
                    "SCFNO2d was created with "
                    "use_condition=True, but no "
                    "condition was provided."
                )

            condition = jnp.asarray(
                c,
                dtype=dtype,
            )

            # (B,H,W) -> (B,H,W,1)
            if (
                condition.ndim
                ==
                3
            ):

                condition = (
                    condition[
                        ...,
                        None
                    ]
                )

            if (
                condition.shape[
                    1
                ]
                !=
                nx
                or
                condition.shape[
                    2
                ]
                !=
                ny
            ):

                raise ValueError(
                    "Condition spatial shape "
                    "does not match the state."
                )

            features.append(
                condition
            )

        # --------------------------------------------------------
        # Continuous physical time
        # --------------------------------------------------------

        if self.use_time:

            if t is None:

                raise ValueError(
                    "SCFNO2d was created with "
                    "use_time=True, but no time "
                    "was provided."
                )

            time_field = (
                self.make_time_field(
                    t,
                    batch_size,
                    nx,
                    ny,
                    dtype,
                )
            )

            features.append(
                time_field
            )

        # ========================================================
        # CONCATENATE
        # ========================================================

        network_input = (
            jnp.concatenate(
                features,
                axis=-1,
            )
        )

        # ========================================================
        # LIFT TO LATENT WIDTH
        # ========================================================

        latent = nn.Dense(
            features=(
                self.width
            ),
            name="lifting",
        )(
            network_input
        )

        latent = nn.gelu(
            latent
        )

        # ========================================================
        # CREATE THE SHARED BLOCK ONCE
        #
        # Calling this same Flax submodule repeatedly means that
        # every composition step shares exactly the same params.
        # ========================================================

        shared_block = SharedFNOBlock(
            width=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            residual_scale=(
                self.residual_scale
            ),
            name="shared_operator",
        )

        # ========================================================
        # SELF-COMPOSITION
        # ========================================================

        for _ in range(
            self.compose_depth
        ):

            latent = shared_block(
                latent
            )

        # ========================================================
        # FINAL NORMALIZATION
        # ========================================================

        latent = nn.LayerNorm(
            name="output_norm"
        )(
            latent
        )

        # ========================================================
        # PROJECTION TO q_t
        # ========================================================

        latent = nn.Dense(
            features=128,
            name="projection_1",
        )(
            latent
        )

        latent = nn.gelu(
            latent
        )

        output = nn.Dense(
            features=(
                self.num_channels
            ),
            name="projection_2",
        )(
            latent
        )

        return output


__all__ = [
    "SpectralConv2d",
    "SharedFNOBlock",
    "SCFNO2d",
]