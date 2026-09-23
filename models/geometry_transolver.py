"""
Geometry-conditioned 2D Transolver backbone for WB-Bed-PI-CFO.

Architecture
------------

Dynamic state:
    q = [h, hu, hv]

Geometry condition:
    b(x, y)

Geometry representation:
    b
      -> [b, b_x, b_y]
      -> GeometryEncoder2d
      -> latent geometry features

Main state representation:
    [q, grid, b]
      -> linear lifting

Fusion:
    lifted state
    +
    geometry latent
    +
    physical time embedding

Operator:
    Transolver Physics-Attention blocks

Output:
    q_t = [h_t, (hu)_t, (hv)_t]

Important
---------

This model replaces ONLY the neural-operator backbone.

It does NOT replace:

    Continuous Flow Operator (CFO)
    quintic spline construction
    conditional flow matching
    SWE PDE loss
    bed-response loss
    well-balanced loss
    RK4 rollout

External interface
------------------

The interface matches the existing geometry-aware backbones:

    model(x, t, c, grid=None)

where

    x = q
    t = physical time
    c = bathymetry
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
# PHYSICS ATTENTION
# =====================================================================

class PhysicsAttention2d(nn.Module):
    """
    Structured-mesh Physics-Attention used by Transolver.

    Instead of performing attention directly between every pair of
    physical grid points, spatial features are softly grouped into
    a smaller number of learned physical slices.

    Pipeline
    --------

        spatial features
              |
              v
        point -> slice weights
              |
              v
        physical slice tokens
              |
              v
        token self-attention
              |
              v
           de-slice
              |
              v
        spatial features

    Input
    -----
    x:
        (B, H, W, C)

    Output
    ------
        (B, H, W, C)
    """

    width: int
    num_heads: int = 4
    slice_num: int = 32
    kernel_size: int = 3


    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        # ---------------------------------------------------------
        # INPUT CHECKS
        # ---------------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "PhysicsAttention2d expects "
                "input shape (B,H,W,C), "
                f"received {x.shape}."
            )

        if self.width <= 0:

            raise ValueError(
                "width must be positive."
            )

        if self.num_heads <= 0:

            raise ValueError(
                "num_heads must be positive."
            )

        if self.slice_num <= 0:

            raise ValueError(
                "slice_num must be positive."
            )

        if (
            self.width
            %
            self.num_heads
            !=
            0
        ):

            raise ValueError(
                "PhysicsAttention2d requires "
                "width % num_heads == 0. "
                f"Got width={self.width}, "
                f"num_heads={self.num_heads}."
            )

        # ---------------------------------------------------------
        # DIMENSIONS
        # ---------------------------------------------------------

        (
            batch_size,
            height,
            width_spatial,
            channels,
        ) = x.shape

        if channels != self.width:

            raise ValueError(
                "PhysicsAttention2d expected "
                f"{self.width} channels, "
                f"received {channels}."
            )

        num_points = (
            height
            *
            width_spatial
        )

        dim_head = (
            self.width
            //
            self.num_heads
        )

        inner_dim = (
            self.num_heads
            *
            dim_head
        )

        # =========================================================
        # STEP 1
        # LOCAL SPATIAL PROJECTIONS
        # =========================================================

        feature_projection = nn.Conv(
            features=inner_dim,
            kernel_size=(
                self.kernel_size,
                self.kernel_size,
            ),
            padding="SAME",
            name="in_project_fx",
        )(
            x
        )

        slice_projection = nn.Conv(
            features=inner_dim,
            kernel_size=(
                self.kernel_size,
                self.kernel_size,
            ),
            padding="SAME",
            name="in_project_x",
        )(
            x
        )

        # ---------------------------------------------------------
        # (B,H,W,C)
        #
        # ->
        #
        # (B,heads,N,dim_head)
        # ---------------------------------------------------------

        feature_projection = (
            feature_projection.reshape(
                batch_size,
                num_points,
                self.num_heads,
                dim_head,
            )
        )

        feature_projection = jnp.transpose(
            feature_projection,
            (
                0,
                2,
                1,
                3,
            ),
        )

        slice_projection = (
            slice_projection.reshape(
                batch_size,
                num_points,
                self.num_heads,
                dim_head,
            )
        )

        slice_projection = jnp.transpose(
            slice_projection,
            (
                0,
                2,
                1,
                3,
            ),
        )

        # =========================================================
        # STEP 2
        # POINT -> PHYSICAL SLICE ASSIGNMENT
        # =========================================================

        slice_logits = nn.Dense(
            features=self.slice_num,
            kernel_init=(
                nn.initializers.orthogonal()
            ),
            name="in_project_slice",
        )(
            slice_projection
        )

        # ---------------------------------------------------------
        # Trainable Transolver temperature
        # ---------------------------------------------------------

        temperature = self.param(
            "temperature",
            nn.initializers.constant(
                0.5
            ),
            (
                1,
                self.num_heads,
                1,
                1,
            ),
        )

        temperature = jnp.clip(
            temperature,
            0.1,
            5.0,
        )

        slice_weights = jnn.softmax(
            slice_logits
            /
            temperature,
            axis=-1,
        )

        # ---------------------------------------------------------
        # Shape:
        #
        # (B, heads, slice_num)
        # ---------------------------------------------------------

        slice_norm = jnp.sum(
            slice_weights,
            axis=2,
        )

        # =========================================================
        # STEP 3
        # BUILD PHYSICAL SLICE TOKENS
        # =========================================================

        slice_tokens = jnp.einsum(
            "bhnc,bhng->bhgc",
            feature_projection,
            slice_weights,
        )

        slice_tokens = (
            slice_tokens
            /
            (
                slice_norm[
                    ...,
                    None,
                ]
                +
                1.0e-5
            )
        )

        # =========================================================
        # STEP 4
        # ATTENTION BETWEEN PHYSICAL SLICE TOKENS
        # =========================================================

        query = nn.Dense(
            features=dim_head,
            use_bias=False,
            name="to_q",
        )(
            slice_tokens
        )

        key = nn.Dense(
            features=dim_head,
            use_bias=False,
            name="to_k",
        )(
            slice_tokens
        )

        value = nn.Dense(
            features=dim_head,
            use_bias=False,
            name="to_v",
        )(
            slice_tokens
        )

        attention_scale = (
            dim_head
            **
            -0.5
        )

        attention_logits = jnp.einsum(
            "bhgd,bhkd->bhgk",
            query,
            key,
        )

        attention_logits = (
            attention_logits
            *
            attention_scale
        )

        attention_weights = jnn.softmax(
            attention_logits,
            axis=-1,
        )

        attended_tokens = jnp.einsum(
            "bhgk,bhkd->bhgd",
            attention_weights,
            value,
        )

        # =========================================================
        # STEP 5
        # DE-SLICE BACK TO PHYSICAL GRID
        # =========================================================

        output = jnp.einsum(
            "bhgd,bhng->bhnd",
            attended_tokens,
            slice_weights,
        )

        output = jnp.transpose(
            output,
            (
                0,
                2,
                1,
                3,
            ),
        )

        output = output.reshape(
            batch_size,
            height,
            width_spatial,
            inner_dim,
        )

        output = nn.Dense(
            features=self.width,
            name="to_out",
        )(
            output
        )

        return output


# =====================================================================
# TRANSOLVER BLOCK
# =====================================================================

class TransolverBlock2d(nn.Module):
    """
    One Transolver block.

    Structure:

        x
        |
      LayerNorm
        |
    PhysicsAttention
        |
       + x
        |
      LayerNorm
        |
       MLP
        |
       + x
    """

    width: int
    num_heads: int = 4
    slice_num: int = 32
    mlp_ratio: int = 2


    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:

        # ---------------------------------------------------------
        # PHYSICS ATTENTION RESIDUAL
        # ---------------------------------------------------------

        attention_input = nn.LayerNorm(
            name="attention_norm",
        )(
            x
        )

        attention_output = PhysicsAttention2d(
            width=self.width,
            num_heads=self.num_heads,
            slice_num=self.slice_num,
            name="physics_attention",
        )(
            attention_input
        )

        x = (
            x
            +
            attention_output
        )

        # ---------------------------------------------------------
        # MLP RESIDUAL
        # ---------------------------------------------------------

        mlp_input = nn.LayerNorm(
            name="mlp_norm",
        )(
            x
        )

        mlp_hidden = nn.Dense(
            features=(
                self.width
                *
                self.mlp_ratio
            ),
            name="mlp_dense_1",
        )(
            mlp_input
        )

        mlp_hidden = jnn.gelu(
            mlp_hidden
        )

        mlp_output = nn.Dense(
            features=self.width,
            name="mlp_dense_2",
        )(
            mlp_hidden
        )

        return (
            x
            +
            mlp_output
        )


# =====================================================================
# GEOMETRY-AWARE TRANSOLVER
# =====================================================================

class GeometryTransolver2d(nn.Module):
    """
    Geometry-conditioned Transolver backbone for Continuous Flow
    Operator training.

    Predicts:

        q_t = N_theta(t, q, b)

    State
    -----
        q = [h, hu, hv]

    Condition
    ---------
        b = bathymetry

    Geometry branch
    ---------------

        b
        ->
        [b, b_x, b_y]
        ->
        GeometryEncoder2d

    Output
    ------

        [
            dh/dt,
            d(hu)/dt,
            d(hv)/dt
        ]
    """

    # -----------------------------------------------------------------
    # OUTPUT CHANNELS
    # -----------------------------------------------------------------

    num_channels: int

    # -----------------------------------------------------------------
    # MODEL WIDTH
    # -----------------------------------------------------------------

    width: int = 64

    # -----------------------------------------------------------------
    # NUMBER OF TRANSOLVER BLOCKS
    # -----------------------------------------------------------------

    num_blocks: int = 4

    # -----------------------------------------------------------------
    # PHYSICS ATTENTION
    # -----------------------------------------------------------------

    num_heads: int = 4

    slice_num: int = 32

    mlp_ratio: int = 2

    # -----------------------------------------------------------------
    # GEOMETRY ENCODER
    # -----------------------------------------------------------------

    geometry_width: int = 16

    geometry_depth: int = 2

    dx: float = 0.15625

    dy: float = 0.15625

    include_gradient_magnitude: bool = False

    # -----------------------------------------------------------------
    # CFO TIME CONDITIONING
    # -----------------------------------------------------------------

    use_time: bool = True


    def setup(
        self,
    ):

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        if self.num_channels <= 0:

            raise ValueError(
                "num_channels must be positive."
            )

        if self.width <= 0:

            raise ValueError(
                "width must be positive."
            )

        if self.num_blocks <= 0:

            raise ValueError(
                "num_blocks must be positive."
            )

        if self.num_heads <= 0:

            raise ValueError(
                "num_heads must be positive."
            )

        if self.slice_num <= 0:

            raise ValueError(
                "slice_num must be positive."
            )

        if self.mlp_ratio <= 0:

            raise ValueError(
                "mlp_ratio must be positive."
            )

        if (
            self.width
            %
            self.num_heads
            !=
            0
        ):

            raise ValueError(
                "GeometryTransolver2d requires "
                "width % num_heads == 0. "
                f"Got width={self.width}, "
                f"num_heads={self.num_heads}."
            )

        # =========================================================
        # GEOMETRY ENCODER
        # =========================================================

        self.geometry_encoder = (
            GeometryEncoder2d(
                output_width=self.width,
                hidden_width=(
                    self.geometry_width
                ),
                depth=(
                    self.geometry_depth
                ),
                dx=self.dx,
                dy=self.dy,
                include_gradient_magnitude=(
                    self.include_gradient_magnitude
                ),
            )
        )

        # =========================================================
        # INPUT LIFT
        #
        # [q, grid, b] -> width
        # =========================================================

        self.input_lift = nn.Dense(
            features=self.width,
            name="input_lift",
        )

        # =========================================================
        # TRANSOLVER BLOCKS
        # =========================================================

        self.transolver_blocks = [

            TransolverBlock2d(
                width=self.width,
                num_heads=(
                    self.num_heads
                ),
                slice_num=(
                    self.slice_num
                ),
                mlp_ratio=(
                    self.mlp_ratio
                ),
                name=(
                    f"transolver_block_{block_index}"
                ),
            )

            for block_index
            in range(
                self.num_blocks
            )
        ]

        # =========================================================
        # OUTPUT NORMALIZATION
        #
        # IMPORTANT:
        #
        # Because this class uses setup(), LayerNorm must also be
        # created here rather than directly inside __call__().
        # =========================================================

        self.output_norm = nn.LayerNorm(
            name="output_norm",
        )

        # =========================================================
        # OUTPUT PROJECTION
        # =========================================================

        self.output_hidden = nn.Dense(
            features=128,
            name="output_hidden",
        )

        self.output_projection = nn.Dense(
            features=self.num_channels,
            name="output_projection",
        )


    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        c: Optional[jnp.ndarray] = None,
        grid: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        # =========================================================
        # STATE SHAPE
        # =========================================================

        if x.ndim == 3:

            x = x[
                ...,
                None,
            ]

        if x.ndim != 4:

            raise ValueError(
                "GeometryTransolver2d expects "
                "state shape (B,H,W,C), "
                f"received {x.shape}."
            )

        # =========================================================
        # GEOMETRY CONDITION
        # =========================================================

        if c is None:

            raise ValueError(
                "GeometryTransolver2d requires "
                "geometry condition c."
            )

        c = ensure_geometry_channels(
            c
        )

        (
            batch_size,
            height,
            width_spatial,
            _,
        ) = x.shape

        if (
            c.shape[0]
            !=
            batch_size
        ):

            raise ValueError(
                "State and geometry batch sizes "
                "must match."
            )

        if (
            c.shape[1]
            !=
            height
            or
            c.shape[2]
            !=
            width_spatial
        ):

            raise ValueError(
                "State and geometry spatial "
                "dimensions must match. "
                f"State={x.shape}, "
                f"geometry={c.shape}."
            )

        # =========================================================
        # COORDINATE GRID
        # =========================================================

        if grid is None:

            grid_x = jnp.linspace(
                0.0,
                1.0,
                height,
                dtype=x.dtype,
            )

            grid_y = jnp.linspace(
                0.0,
                1.0,
                width_spatial,
                dtype=x.dtype,
            )

            xx, yy = jnp.meshgrid(
                grid_x,
                grid_y,
                indexing="ij",
            )

            grid = jnp.stack(
                [
                    xx,
                    yy,
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
                    width_spatial,
                    2,
                ),
            )

        else:

            if grid.ndim == 3:

                grid = jnp.broadcast_to(
                    grid[
                        None,
                        ...
                    ],
                    (
                        batch_size,
                        height,
                        width_spatial,
                        grid.shape[-1],
                    ),
                )

            if grid.ndim != 4:

                raise ValueError(
                    "grid must have shape "
                    "(H,W,C) or (B,H,W,C)."
                )

            if (
                grid.shape[0]
                !=
                batch_size
                or
                grid.shape[1]
                !=
                height
                or
                grid.shape[2]
                !=
                width_spatial
            ):

                raise ValueError(
                    "Provided grid does not match "
                    "state dimensions."
                )

        # =========================================================
        # MAIN INPUT
        #
        # [h, hu, hv, x, y, b]
        # =========================================================

        lifted_input = jnp.concatenate(
            [
                x,
                grid,
                c,
            ],
            axis=-1,
        )

        latent = self.input_lift(
            lifted_input
        )

        # =========================================================
        # GEOMETRY LATENT
        #
        # b -> [b, bx, by] -> CNN
        # =========================================================

        geometry_latent = (
            self.geometry_encoder(
                c
            )
        )

        latent = (
            latent
            +
            geometry_latent
        )

        # =========================================================
        # PHYSICAL TIME FOR CFO
        # =========================================================

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

            latent = (
                latent
                +
                time_embedding[
                    :,
                    None,
                    None,
                    :,
                ]
            )

        # =========================================================
        # TRANSOLVER BACKBONE
        # =========================================================

        for block in (
            self.transolver_blocks
        ):

            latent = block(
                latent
            )

        # =========================================================
        # OUTPUT q_t
        # =========================================================

        latent = self.output_norm(
            latent
        )

        latent = self.output_hidden(
            latent
        )

        latent = jnn.gelu(
            latent
        )

        output = self.output_projection(
            latent
        )

        return output


__all__ = [
    "PhysicsAttention2d",
    "TransolverBlock2d",
    "GeometryTransolver2d",
]