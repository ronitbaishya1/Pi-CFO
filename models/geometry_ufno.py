"""
Geometry-conditioned 2D U-FNO-style backbone.

Architecture
------------
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


Cross-resolution inference
--------------------------
For a model trained at 64x64 and evaluated at 128x128, the local
U-Net branch can optionally be evaluated at its original 64x64
resolution.

Example:

    128 latent
        |
        v
    area downsample
        |
        v
    64 latent
        |
        v
    trained U-Net
        |
        v
    interpolation to 128
        |
        v
    combine with 128 Fourier branch

The reconstruction method can be selected:

    linear
    cubic
    lanczos3

An optional small contribution from the native-resolution U-Net
response may also be blended into the canonical response.

No additional learned parameters are introduced.

If all cross-resolution options are left at their defaults, this
class behaves like the original Geometry-U-FNO.
"""

from __future__ import annotations

from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp

from jax import image as jimage

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
# RESOLUTION HELPERS
# =====================================================================

VALID_RESIZE_METHODS = {
    "linear",
    "cubic",
    "lanczos3",
    "lanczos5",
}


def downsample_to_resolution(
    x: jnp.ndarray,
    target_height: int,
    target_width: int,
    *,
    method: str = "area",
) -> jnp.ndarray:
    """
    Downsample a spatial field.

    method="area"
    -------------
    For integer ratios such as 128 -> 64, perform exact block
    averaging.

    This is the preferred anti-aliased downsampling method for the
    present zero-shot experiment.

    Other supported methods:
        linear
        cubic
        lanczos3
        lanczos5
    """

    (
        batch,
        height,
        width,
        channels,
    ) = x.shape

    target_height = int(
        target_height
    )

    target_width = int(
        target_width
    )

    # -----------------------------------------------------------------
    # NO RESIZE NEEDED
    # -----------------------------------------------------------------

    if (
        height == target_height
        and
        width == target_width
    ):

        return x

    # -----------------------------------------------------------------
    # AREA / BLOCK AVERAGE
    # -----------------------------------------------------------------

    if method == "area":

        if (
            height >= target_height
            and
            width >= target_width
            and
            height % target_height == 0
            and
            width % target_width == 0
        ):

            scale_h = (
                height
                //
                target_height
            )

            scale_w = (
                width
                //
                target_width
            )

            reshaped = jnp.reshape(
                x,
                (
                    batch,
                    target_height,
                    scale_h,
                    target_width,
                    scale_w,
                    channels,
                ),
            )

            return jnp.mean(
                reshaped,
                axis=(
                    2,
                    4,
                ),
            )

        # -------------------------------------------------------------
        # GENERAL FALLBACK IF THE RATIO IS NOT INTEGER
        # -------------------------------------------------------------

        return jimage.resize(
            x,
            (
                batch,
                target_height,
                target_width,
                channels,
            ),
            method="linear",
        )

    # -----------------------------------------------------------------
    # JAX INTERPOLATION
    # -----------------------------------------------------------------

    if method not in VALID_RESIZE_METHODS:

        raise ValueError(
            "Unknown downsample method "
            f"'{method}'."
        )

    return jimage.resize(
        x,
        (
            batch,
            target_height,
            target_width,
            channels,
        ),
        method=method,
    )


def upsample_to_resolution(
    x: jnp.ndarray,
    target_height: int,
    target_width: int,
    *,
    method: str = "linear",
) -> jnp.ndarray:
    """
    Upsample a spatial field.

    Supported methods:
        linear
        cubic
        lanczos3
        lanczos5
    """

    (
        batch,
        height,
        width,
        channels,
    ) = x.shape

    target_height = int(
        target_height
    )

    target_width = int(
        target_width
    )

    if (
        height == target_height
        and
        width == target_width
    ):

        return x

    if method not in VALID_RESIZE_METHODS:

        raise ValueError(
            "Unknown upsample method "
            f"'{method}'."
        )

    return jimage.resize(
        x,
        (
            batch,
            target_height,
            target_width,
            channels,
        ),
        method=method,
    )


# =====================================================================
# OPTIONAL SOURCE-RESOLUTION SPECTRAL FILTER
#
# Kept for previous experiments.
# It will remain OFF in the current resampling comparison.
# =====================================================================

def _frequency_taper(
    size: int,
    source_resolution: int,
    taper_start: float,
    dtype,
) -> jnp.ndarray:

    if source_resolution >= size:

        return jnp.ones(
            size,
            dtype=dtype,
        )

    frequency = jnp.abs(
        jnp.fft.fftfreq(
            size
        )
    )

    source_nyquist = (
        0.5
        *
        float(
            source_resolution
        )
        /
        float(
            size
        )
    )

    passband = (
        float(
            taper_start
        )
        *
        source_nyquist
    )

    denominator = max(
        source_nyquist
        -
        passband,
        1.0e-12,
    )

    transition = jnp.clip(
        (
            frequency
            -
            passband
        )
        /
        denominator,
        0.0,
        1.0,
    )

    cosine_taper = (
        0.5
        *
        (
            1.0
            +
            jnp.cos(
                jnp.pi
                *
                transition
            )
        )
    )

    weights = jnp.where(
        frequency
        <=
        passband,
        1.0,
        cosine_taper,
    )

    weights = jnp.where(
        frequency
        >=
        source_nyquist,
        0.0,
        weights,
    )

    return weights.astype(
        dtype
    )


def source_resolution_projection(
    field: jnp.ndarray,
    *,
    source_resolution: int,
    taper_start: float = 0.85,
) -> jnp.ndarray:

    (
        _,
        height,
        width,
        _,
    ) = field.shape

    source_resolution = int(
        source_resolution
    )

    if (
        height <= source_resolution
        and
        width <= source_resolution
    ):

        return field

    wx = _frequency_taper(
        height,
        source_resolution,
        taper_start,
        field.dtype,
    )

    wy = _frequency_taper(
        width,
        source_resolution,
        taper_start,
        field.dtype,
    )

    window = (
        wx[
            None,
            :,
            None,
            None,
        ]
        *
        wy[
            None,
            None,
            :,
            None,
        ]
    )

    field_fft = jnp.fft.fft2(
        field,
        axes=(
            1,
            2,
        ),
        norm="ortho",
    )

    filtered_fft = (
        field_fft
        *
        window
    )

    filtered = jnp.fft.ifft2(
        filtered_fft,
        axes=(
            1,
            2,
        ),
        norm="ortho",
    ).real

    return filtered.astype(
        field.dtype
    )


# =====================================================================
# MINI U-NET
# =====================================================================

class MiniUNet2d(nn.Module):

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

        # -----------------------------------------------------------------
        # LEVEL 0
        # -----------------------------------------------------------------

        x0 = nn.Conv(
            self.width,
            kernel_size=(
                3,
                3,
            ),
            padding="SAME",
            name="input_conv",
        )(
            x
        )

        x0 = jnn.gelu(
            x0
        )

        # -----------------------------------------------------------------
        # DOWN 1
        # -----------------------------------------------------------------

        d1 = nn.Conv(
            self.width,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                2,
                2,
            ),
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
            kernel_size=(
                3,
                3,
            ),
            padding="SAME",
            name="down_1_refine",
        )(
            d1
        )

        d1 = jnn.gelu(
            d1
        )

        # -----------------------------------------------------------------
        # DOWN 2
        # -----------------------------------------------------------------

        d2 = nn.Conv(
            self.width,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                2,
                2,
            ),
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
            kernel_size=(
                3,
                3,
            ),
            padding="SAME",
            name="bottleneck",
        )(
            d2
        )

        d2 = jnn.gelu(
            d2
        )

        # -----------------------------------------------------------------
        # UP 1
        # -----------------------------------------------------------------

        u1 = nn.ConvTranspose(
            self.width,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                2,
                2,
            ),
            padding="SAME",
            name="up_1",
        )(
            d2
        )

        if (
            u1.shape[
                1:
                3
            ]
            !=
            d1.shape[
                1:
                3
            ]
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

        # -----------------------------------------------------------------
        # UP 2
        # -----------------------------------------------------------------

        u2 = nn.ConvTranspose(
            self.width,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                2,
                2,
            ),
            padding="SAME",
            name="up_2",
        )(
            u1
        )

        if (
            u2.shape[
                1:
                3
            ]
            !=
            x0.shape[
                1:
                3
            ]
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
            kernel_size=(
                1,
                1,
            ),
            padding="SAME",
            name="output_projection",
        )(
            u2
        )


# =====================================================================
# GEOMETRY U-FNO
# =====================================================================

class GeometryUFNO2d(nn.Module):

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

    # -----------------------------------------------------------------
    # CANONICAL LOCAL BRANCH
    # -----------------------------------------------------------------

    canonical_unet_resolution: Optional[int] = None

    canonical_geometry_resolution: Optional[int] = None

    canonical_downsample_mode: str = "area"

    canonical_upsample_mode: str = "linear"

    # -----------------------------------------------------------------
    # OPTIONAL NATIVE-U-NET BLEND
    #
    # 0.0 = pure canonical response
    # 0.1 = 90% canonical + 10% native response
    # -----------------------------------------------------------------

    canonical_native_blend: float = 0.0

    # -----------------------------------------------------------------
    # OPTIONAL FINAL VECTOR-FIELD FILTER
    # -----------------------------------------------------------------

    source_resolution_filter: Optional[int] = None

    filter_taper_start: float = 0.85

    # -----------------------------------------------------------------
    # PHYSICAL DOMAIN
    # -----------------------------------------------------------------

    domain_length_x: float = 5.0
    domain_length_y: float = 5.0

    # =================================================================
    # SETUP
    # =================================================================

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

        # -----------------------------------------------------------------
        # CANONICAL U-NET VALIDATION
        # -----------------------------------------------------------------

        if (
            self.canonical_unet_resolution
            is not None
        ):

            canonical = int(
                self.canonical_unet_resolution
            )

            if canonical < 4:

                raise ValueError(
                    "canonical_unet_resolution "
                    "must be >= 4."
                )

            if (
                canonical
                %
                4
                !=
                0
            ):

                raise ValueError(
                    "canonical_unet_resolution "
                    "must be divisible by 4."
                )

        # -----------------------------------------------------------------
        # RESAMPLING VALIDATION
        # -----------------------------------------------------------------

        valid_downsample = {
            "area",
            *VALID_RESIZE_METHODS,
        }

        if (
            self.canonical_downsample_mode
            not in
            valid_downsample
        ):

            raise ValueError(
                "Invalid canonical_downsample_mode: "
                f"{self.canonical_downsample_mode}"
            )

        if (
            self.canonical_upsample_mode
            not in
            VALID_RESIZE_METHODS
        ):

            raise ValueError(
                "Invalid canonical_upsample_mode: "
                f"{self.canonical_upsample_mode}"
            )

        if not (
            0.0
            <=
            float(
                self.canonical_native_blend
            )
            <=
            1.0
        ):

            raise ValueError(
                "canonical_native_blend must "
                "be between 0 and 1."
            )

        # -----------------------------------------------------------------
        # GEOMETRY SPACING
        # -----------------------------------------------------------------

        geometry_dx = float(
            self.dx
        )

        geometry_dy = float(
            self.dy
        )

        if (
            self.canonical_geometry_resolution
            is not None
        ):

            canonical_geometry = int(
                self.canonical_geometry_resolution
            )

            geometry_dx = (
                float(
                    self.domain_length_x
                )
                /
                float(
                    canonical_geometry
                )
            )

            geometry_dy = (
                float(
                    self.domain_length_y
                )
                /
                float(
                    canonical_geometry
                )
            )

        # -----------------------------------------------------------------
        # GEOMETRY ENCODER
        # -----------------------------------------------------------------

        self.geometry_encoder = (
            GeometryEncoder2d(
                output_width=(
                    self.width
                ),

                hidden_width=(
                    self.geometry_width
                ),

                depth=(
                    self.geometry_depth
                ),

                dx=(
                    geometry_dx
                ),

                dy=(
                    geometry_dy
                ),

                include_gradient_magnitude=(
                    self.include_gradient_magnitude
                ),
            )
        )

        # -----------------------------------------------------------------
        # INPUT LIFTING
        # -----------------------------------------------------------------

        self.fc0 = nn.Dense(
            self.width
        )

        # -----------------------------------------------------------------
        # FOURIER BLOCKS
        # -----------------------------------------------------------------

        self.convs = [
            SpectralConv2d(
                self.width,
                self.width,
                self.modes1,
                self.modes2,
            )

            for _
            in range(
                self.num_blocks
            )
        ]

        # -----------------------------------------------------------------
        # POINTWISE BRANCH
        # -----------------------------------------------------------------

        self.ws = [
            nn.Conv(
                self.width,
                kernel_size=(
                    1,
                    1,
                ),
            )

            for _
            in range(
                self.num_blocks
            )
        ]

        # -----------------------------------------------------------------
        # U-NET BRANCH
        # -----------------------------------------------------------------

        self.unets = [
            MiniUNet2d(
                self.width
            )

            for _
            in range(
                self.num_u_blocks
            )
        ]

        # -----------------------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------------------

        self.fc1 = nn.Dense(
            128
        )

        self.fc2 = nn.Dense(
            self.num_channels
        )

    # =================================================================
    # GEOMETRY ENCODER
    # =================================================================

    def _encode_geometry(
        self,
        c,
        target_height,
        target_width,
    ):

        canonical = (
            self.canonical_geometry_resolution
        )

        # -----------------------------------------------------------------
        # NATIVE TARGET GRID
        # -----------------------------------------------------------------

        if canonical is None:

            return self.geometry_encoder(
                c
            )

        canonical = int(
            canonical
        )

        if (
            c.shape[1] == canonical
            and
            c.shape[2] == canonical
        ):

            return self.geometry_encoder(
                c
            )

        # -----------------------------------------------------------------
        # TARGET -> CANONICAL
        # -----------------------------------------------------------------

        c_canonical = (
            downsample_to_resolution(
                c,
                canonical,
                canonical,
                method=(
                    self.canonical_downsample_mode
                ),
            )
        )

        # -----------------------------------------------------------------
        # GEOMETRY CNN
        # -----------------------------------------------------------------

        geometry_latent = (
            self.geometry_encoder(
                c_canonical
            )
        )

        # -----------------------------------------------------------------
        # CANONICAL -> TARGET
        # -----------------------------------------------------------------

        geometry_latent = (
            upsample_to_resolution(
                geometry_latent,
                target_height,
                target_width,
                method=(
                    self.canonical_upsample_mode
                ),
            )
        )

        return geometry_latent

    # =================================================================
    # U-NET BRANCH
    # =================================================================

    def _apply_unet_branch(
        self,
        unet,
        z,
    ):

        (
            _,
            height,
            width,
            _,
        ) = z.shape

        canonical = (
            self.canonical_unet_resolution
        )

        # -----------------------------------------------------------------
        # ORIGINAL NATIVE-RESOLUTION U-NET
        # -----------------------------------------------------------------

        if canonical is None:

            return unet(
                z
            )

        canonical = int(
            canonical
        )

        if (
            height == canonical
            and
            width == canonical
        ):

            return unet(
                z
            )

        # -----------------------------------------------------------------
        # TARGET -> CANONICAL
        # -----------------------------------------------------------------

        z_canonical = (
            downsample_to_resolution(
                z,
                canonical,
                canonical,
                method=(
                    self.canonical_downsample_mode
                ),
            )
        )

        # -----------------------------------------------------------------
        # U-NET AT TRAINING RESOLUTION
        # -----------------------------------------------------------------

        canonical_response = (
            unet(
                z_canonical
            )
        )

        # -----------------------------------------------------------------
        # CANONICAL -> TARGET
        # -----------------------------------------------------------------

        canonical_response = (
            upsample_to_resolution(
                canonical_response,
                height,
                width,
                method=(
                    self.canonical_upsample_mode
                ),
            )
        )

        # -----------------------------------------------------------------
        # OPTIONAL SMALL NATIVE-RESOLUTION CONTRIBUTION
        # -----------------------------------------------------------------

        blend = float(
            self.canonical_native_blend
        )

        if blend <= 0.0:

            return canonical_response

        native_response = (
            unet(
                z
            )
        )

        return (
            (
                1.0
                -
                blend
            )
            *
            canonical_response
            +
            blend
            *
            native_response
        )

    # =================================================================
    # FORWARD
    # =================================================================

    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        c: Optional[jnp.ndarray] = None,
        grid: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        if x.ndim == 3:

            x = x[
                ...,
                None
            ]

        if c is None:

            raise ValueError(
                "GeometryUFNO2d requires geometry c."
            )

        c = ensure_geometry_channels(
            c
        )

        (
            batch_size,
            height,
            width,
            _,
        ) = x.shape

        # -----------------------------------------------------------------
        # NORMALIZED GRID
        # -----------------------------------------------------------------

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
                grid[
                    None,
                    ...
                ],
                (
                    batch_size,
                    1,
                    1,
                    1,
                ),
            )

        # -----------------------------------------------------------------
        # LIFT INPUT
        # -----------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # GEOMETRY CONDITIONING
        # -----------------------------------------------------------------

        geometry_latent = (
            self._encode_geometry(
                c,
                height,
                width,
            )
        )

        z = (
            z
            +
            geometry_latent
        )

        # -----------------------------------------------------------------
        # TIME
        # -----------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # FIRST U-NET BLOCK
        # -----------------------------------------------------------------

        u_start = (
            self.num_blocks
            -
            self.num_u_blocks
        )

        # -----------------------------------------------------------------
        # OPERATOR BLOCKS
        # -----------------------------------------------------------------

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

            # -------------------------------------------------------------
            # U-NET CONTRIBUTION
            # -------------------------------------------------------------

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

                unet_response = (
                    self._apply_unet_branch(
                        self.unets[
                            unet_index
                        ],
                        z,
                    )
                )

                block_output = (
                    block_output
                    +
                    unet_response
                )

            # -------------------------------------------------------------
            # ACTIVATION
            # -------------------------------------------------------------

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

        # -----------------------------------------------------------------
        # OUTPUT PROJECTION
        # -----------------------------------------------------------------

        z = jnn.gelu(
            self.fc1(
                z
            )
        )

        output = (
            self.fc2(
                z
            )
        )

        # -----------------------------------------------------------------
        # OPTIONAL SOURCE FILTER
        # -----------------------------------------------------------------

        if (
            self.source_resolution_filter
            is not None
        ):

            output = (
                source_resolution_projection(
                    output,

                    source_resolution=(
                        int(
                            self.source_resolution_filter
                        )
                    ),

                    taper_start=(
                        self.filter_taper_start
                    ),
                )
            )

        return output


__all__ = [
    "MiniUNet2d",
    "GeometryUFNO2d",
    "downsample_to_resolution",
    "upsample_to_resolution",
    "source_resolution_projection",
]