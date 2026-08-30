"""DiT model implementation (version B style)."""

from __future__ import annotations

import math
from typing import Any, Optional

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    def timestep_embedding(self, t: jnp.ndarray, max_period: int = 10000) -> jnp.ndarray:
        t = jax.lax.convert_element_type(t, jnp.float32)
        t = t * max_period
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class LabelEmbedder(nn.Module):
    """Convolutional embedder for conditioning label images."""

    hidden_size: int

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        x = y
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        return nn.Dense(features=self.hidden_size)(x)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Any = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.0
    kernel_init: Any = nn.initializers.xavier_uniform()
    bias_init: Any = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        x = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return x


class PatchEmbed(nn.Module):
    """2D image to patch embedding (supports rectangular inputs)."""

    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        _, h, w, _ = x.shape
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=self.bias,
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        patches_h = h // self.patch_size
        patches_w = w // self.patch_size
        return rearrange(x, "b ph pw c -> b (ph pw) c", ph=patches_h, pw=patches_w)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: jnp.ndarray) -> jnp.ndarray:
    """1D sine-cosine positional embedding from positions."""

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    return jnp.concatenate([emb_sin, emb_cos], axis=1)


def get_2d_sincos_pos_embed_rectangle(embed_dim: int, patches_h: int, patches_w: int) -> jnp.ndarray:
    """Return (1, patches_h*patches_w, embed_dim) 2D sin-cos positional embedding."""

    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    grid_h = jnp.arange(patches_h, dtype=jnp.float32)
    grid_w = jnp.arange(patches_w, dtype=jnp.float32)
    gh, gw = jnp.meshgrid(grid_h, grid_w, indexing="ij")

    gh = gh.reshape(-1)
    gw = gw.reshape(-1)

    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, gh)
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, gw)

    emb = jnp.concatenate([emb_h, emb_w], axis=1)
    return emb[None, ...]


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Adaptive layer-norm modulation."""

    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """A single DiT transformer block with adaLN-zero conditioning."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        c = nn.silu(c)
        c = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.0))(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)

        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads,
        )(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class FinalLayer(nn.Module):
    """Final projection layer from tokens back to patch pixels."""

    patch_size: int
    out_channels: int
    hidden_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0.0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)

        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x = modulate(x, shift, scale)
        return nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=nn.initializers.constant(0.0),
        )(x)


class DiT(nn.Module):
    """Diffusion model with Transformer backbone (rectangular input aware)."""

    patch_size: int = 8
    hidden_size: int = 384
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: float = 6.0
    learn_sigma: bool = False
    use_condition: bool = False
    out_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        squeeze_channel = False
        if x.ndim == 3:
            x = x[..., None]
            squeeze_channel = True

        b, h, w, c = x.shape
        out_channels = self.out_channels if self.out_channels is not None else c

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError("Input spatial dimensions must be divisible by patch_size.")

        patches_h = h // self.patch_size
        patches_w = w // self.patch_size

        def init_2d_pos_embed(_: jnp.ndarray) -> jnp.ndarray:
            return get_2d_sincos_pos_embed_rectangle(self.hidden_size, patches_h, patches_w)

        pos_embed = self.param("pos_embed", init_2d_pos_embed)
        pos_embed = jax.lax.stop_gradient(pos_embed)

        if t is not None:
            t_emb = TimestepEmbedder(self.hidden_size)(t)
        else:
            t_emb = jnp.zeros((b, self.hidden_size), dtype=jnp.float32)

        if self.use_condition and y is not None:
            y_emb = LabelEmbedder(self.hidden_size)(y)
            cond = t_emb + y_emb
        else:
            cond = t_emb

        x = PatchEmbed(self.patch_size, self.hidden_size)(x)
        x = x + pos_embed

        for _ in range(self.depth):
            x = DiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio)(x, cond)

        x = FinalLayer(self.patch_size, out_channels, self.hidden_size)(x, cond)
        x = jnp.reshape(x, (b, patches_h, patches_w, self.patch_size, self.patch_size, out_channels))
        x = rearrange(x, "b ph pw hp wp c -> b (ph hp) (pw wp) c")

        if squeeze_channel and out_channels == 1:
            return x[..., 0]
        return x


__all__ = [
    "TimestepEmbedder",
    "LabelEmbedder",
    "MlpBlock",
    "PatchEmbed",
    "DiTBlock",
    "FinalLayer",
    "DiT",
]
