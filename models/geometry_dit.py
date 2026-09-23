"""
Geometry-aware DiT-style backbone for WB-Bed-PI-CFO.

The model predicts the continuous vector field

    q_t = N_theta(t, q, b)

for the shallow-water state

    q = [h, hu, hv]

conditioned on bathymetry b(x,y).

DiT is used only as the neural-operator backbone.

The following remain unchanged:

    Continuous Flow Operator (CFO)
    quintic spline construction
    SWE PDE loss
    bed-response loss
    well-balanced loss
    RK4 rollout
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from einops import rearrange
from flax import linen as nn

from models.dit import (
    DiTBlock,
    FinalLayer,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed_rectangle,
)

from models.geometry_encoder import (
    GeometryEncoder2d,
    ensure_geometry_channels,
)


# =====================================================================
# PATCH-AVERAGE GEOMETRY FEATURES
# =====================================================================

def _patch_average_pool(
    x: jnp.ndarray,
    patch_size: int,
) -> jnp.ndarray:
    """
    Average-pool a spatial feature field into non-overlapping patches.

    Input
    -----
        (B,H,W,C)

    Output
    ------
        (B,N,C)

    where

        N = (H / patch_size) * (W / patch_size)
    """

    if x.ndim != 4:

        raise ValueError(
            "_patch_average_pool expects "
            "input shape (B,H,W,C)."
        )

    (
        batch_size,
        height,
        width,
        channels,
    ) = x.shape

    if (
        height % patch_size != 0
        or
        width % patch_size != 0
    ):

        raise ValueError(
            "Spatial dimensions must be "
            "divisible by patch_size."
        )

    patches_h = (
        height
        //
        patch_size
    )

    patches_w = (
        width
        //
        patch_size
    )

    x = x.reshape(
        batch_size,
        patches_h,
        patch_size,
        patches_w,
        patch_size,
        channels,
    )

    x = jnp.mean(
        x,
        axis=(
            2,
            4,
        ),
    )

    return x.reshape(
        batch_size,
        patches_h
        *
        patches_w,
        channels,
    )


# =====================================================================
# GEOMETRY-DIT
# =====================================================================

class GeometryDiT2d(nn.Module):
    """
    Geometry-conditioned DiT-style transformer for continuous-time CFO.

    State branch
    ------------

        [q, grid, b]
            |
            v
        PatchEmbed
            |
            v
        state tokens


    Geometry branch
    ---------------

        b
        |
        v
    [b, b_x, b_y]
        |
        v
    GeometryEncoder2d
        |
        v
    patch geometry tokens


    Conditioning branch
    -------------------

        physical time
             |
             v
      TimestepEmbedder

             +

       global geometry
             |
             v
         small MLP

             |
             v

        adaLN condition


    Transformer
    -----------

        state tokens
             +
        geometry tokens
             +
      fixed 2D positions
             |
             v
        DiT blocks
             |
             v
      patch-wise q_t
             |
             v
    full-resolution q_t


    IMPORTANT
    ---------

    The positional embedding is NOT a trainable parameter.

    Therefore the learned parameter tree is independent of the
    spatial resolution, enabling later zero-shot 32 -> 64 tests.
    """

    # -----------------------------------------------------------------
    # SWE OUTPUT CHANNELS
    # -----------------------------------------------------------------

    num_channels: int

    # -----------------------------------------------------------------
    # DIT
    # -----------------------------------------------------------------

    patch_size: int = 4

    hidden_size: int = 256

    depth: int = 4

    num_heads: int = 4

    mlp_ratio: float = 4.0

    # -----------------------------------------------------------------
    # GEOMETRY ENCODER
    # -----------------------------------------------------------------

    geometry_width: int = 16

    geometry_depth: int = 2

    dx: float = 0.15625

    dy: float = 0.15625

    include_gradient_magnitude: bool = False

    # -----------------------------------------------------------------
    # CFO PHYSICAL TIME
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

        if self.patch_size <= 0:

            raise ValueError(
                "patch_size must be positive."
            )

        if self.hidden_size <= 0:

            raise ValueError(
                "hidden_size must be positive."
            )

        if self.depth <= 0:

            raise ValueError(
                "depth must be positive."
            )

        if self.num_heads <= 0:

            raise ValueError(
                "num_heads must be positive."
            )

        if (
            self.hidden_size
            %
            self.num_heads
            !=
            0
        ):

            raise ValueError(
                "GeometryDiT2d requires "
                "hidden_size % num_heads == 0."
            )

        # Required by our 2D sin/cos positional representation.
        if (
            self.hidden_size
            %
            4
            !=
            0
        ):

            raise ValueError(
                "GeometryDiT2d requires hidden_size "
                "to be divisible by 4."
            )

        if self.mlp_ratio <= 0:

            raise ValueError(
                "mlp_ratio must be positive."
            )

        # =========================================================
        # GEOMETRY ENCODER
        #
        # b
        # ->
        # [b,b_x,b_y]
        # ->
        # learned spatial features
        # =========================================================

        self.geometry_encoder = (
            GeometryEncoder2d(
                output_width=(
                    self.hidden_size
                ),
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
        # STATE PATCH EMBEDDING
        #
        # Input:
        #
        # [h,hu,hv,x,y,b]
        # =========================================================

        self.state_patch_embed = (
            PatchEmbed(
                patch_size=(
                    self.patch_size
                ),
                embed_dim=(
                    self.hidden_size
                ),
                name="state_patch_embed",
            )
        )

        # =========================================================
        # PHYSICAL TIME EMBEDDING
        # =========================================================

        self.time_embedder = (
            TimestepEmbedder(
                hidden_size=(
                    self.hidden_size
                ),
                name="time_embedder",
            )
        )

        # =========================================================
        # GLOBAL GEOMETRY CONDITION
        # =========================================================

        self.geometry_condition_1 = (
            nn.Dense(
                self.hidden_size,
                name="geometry_condition_1",
            )
        )

        self.geometry_condition_2 = (
            nn.Dense(
                self.hidden_size,
                name="geometry_condition_2",
            )
        )

        # =========================================================
        # DIT TRANSFORMER BLOCKS
        # =========================================================

        self.blocks = [

            DiTBlock(
                hidden_size=(
                    self.hidden_size
                ),
                num_heads=(
                    self.num_heads
                ),
                mlp_ratio=(
                    self.mlp_ratio
                ),
                name=(
                    f"dit_block_{block_index}"
                ),
            )

            for block_index
            in range(
                self.depth
            )
        ]

        # =========================================================
        # PATCH -> q_t
        # =========================================================

        self.final_layer = (
            FinalLayer(
                patch_size=(
                    self.patch_size
                ),
                out_channels=(
                    self.num_channels
                ),
                hidden_size=(
                    self.hidden_size
                ),
                name="final_layer",
            )
        )


    def __call__(
        self,
        x: jnp.ndarray,
        t: Optional[jnp.ndarray] = None,
        c: Optional[jnp.ndarray] = None,
        grid: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:

        # =========================================================
        # STATE
        # =========================================================

        if x.ndim == 3:

            x = x[
                ...,
                None,
            ]

        if x.ndim != 4:

            raise ValueError(
                "GeometryDiT2d expects state "
                "shape (B,H,W,C), "
                f"received {x.shape}."
            )

        # =========================================================
        # BATHYMETRY
        # =========================================================

        if c is None:

            raise ValueError(
                "GeometryDiT2d requires "
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
            or
            c.shape[1]
            !=
            height
            or
            c.shape[2]
            !=
            width_spatial
        ):

            raise ValueError(
                "State and geometry shapes "
                "must match. "
                f"State={x.shape}, "
                f"geometry={c.shape}."
            )

        if (
            height
            %
            self.patch_size
            !=
            0
            or
            width_spatial
            %
            self.patch_size
            !=
            0
        ):

            raise ValueError(
                "State dimensions must be "
                "divisible by patch_size. "
                f"H={height}, "
                f"W={width_spatial}, "
                f"patch_size={self.patch_size}."
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
                        grid.shape[
                            -1
                        ],
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
                    "Provided grid does not "
                    "match state dimensions."
                )

        # =========================================================
        # STATE TOKENS
        #
        # [h,hu,hv,x,y,b]
        # ->
        # patch tokens
        # =========================================================

        state_input = jnp.concatenate(
            [
                x,
                grid,
                c,
            ],
            axis=-1,
        )

        state_tokens = (
            self.state_patch_embed(
                state_input
            )
        )

        # =========================================================
        # GEOMETRY TOKENS
        #
        # GeometryEncoder2d internally builds:
        #
        # [b,b_x,b_y]
        # =========================================================

        geometry_latent = (
            self.geometry_encoder(
                c
            )
        )

        geometry_tokens = (
            _patch_average_pool(
                geometry_latent,
                self.patch_size,
            )
        )

        tokens = (
            state_tokens
            +
            geometry_tokens
        )

        # =========================================================
        # FIXED 2D POSITIONAL EMBEDDING
        #
        # NOT A TRAINABLE PARAMETER.
        # =========================================================

        patches_h = (
            height
            //
            self.patch_size
        )

        patches_w = (
            width_spatial
            //
            self.patch_size
        )

        position_embedding = (
            get_2d_sincos_pos_embed_rectangle(
                self.hidden_size,
                patches_h,
                patches_w,
            )
        )

        position_embedding = (
            position_embedding.astype(
                tokens.dtype
            )
        )

        tokens = (
            tokens
            +
            position_embedding
        )

        # =========================================================
        # PHYSICAL TIME CONDITION
        # =========================================================

        if (
            self.use_time
            and
            t is not None
        ):

            time_condition = (
                self.time_embedder(
                    t
                )
            )

        else:

            time_condition = jnp.zeros(
                (
                    batch_size,
                    self.hidden_size,
                ),
                dtype=tokens.dtype,
            )

        # =========================================================
        # GLOBAL GEOMETRY CONDITION
        #
        # This conditions the adaLN layers in the DiT blocks.
        # =========================================================

        geometry_global = jnp.mean(
            geometry_tokens,
            axis=1,
        )

        geometry_condition = (
            self.geometry_condition_1(
                geometry_global
            )
        )

        geometry_condition = nn.silu(
            geometry_condition
        )

        geometry_condition = (
            self.geometry_condition_2(
                geometry_condition
            )
        )

        condition = (
            time_condition
            +
            geometry_condition
        )

        # =========================================================
        # DIT TRANSFORMER
        # =========================================================

        for block in self.blocks:

            tokens = block(
                tokens,
                condition,
            )

        # =========================================================
        # VECTOR FIELD q_t
        # =========================================================

        output_tokens = (
            self.final_layer(
                tokens,
                condition,
            )
        )

        output = output_tokens.reshape(
            batch_size,
            patches_h,
            patches_w,
            self.patch_size,
            self.patch_size,
            self.num_channels,
        )

        output = rearrange(
            output,
            (
                "b ph pw hp wp c "
                "-> b (ph hp) (pw wp) c"
            ),
        )

        return output


__all__ = [
    "GeometryDiT2d",
]