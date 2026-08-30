"""U-Net family models for CFO experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from flax import linen as nn


def sinusoidal_time_embedding(t: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    """Create sinusoidal time embeddings."""
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half_dim))
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class UNet1D(nn.Module):
    """1D U-Net with optional time and condition inputs."""

    channel_mult: Tuple[int, ...] = (64, 128)
    time_emb_dim: int = 64
    use_condition: bool = False

    class TimeEmbeddingMLP(nn.Module):
        """Project sinusoidal time features."""

        embed_dim: int
        hidden_dim: int = 128

        @nn.compact
        def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
            x = sinusoidal_time_embedding(t, self.embed_dim)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.embed_dim)(x)
            return x

    class DownsampleBlock(nn.Module):
        """1D residual downsampling block."""

        features: int
        use_time_emb: bool = True

        @nn.compact
        def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            x = nn.Conv(features=self.features, kernel_size=(3,), strides=(1,), padding="same")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            if self.use_time_emb and t_emb is not None:
                t_out = nn.Dense(self.features)(t_emb)
                x = x + t_out[:, None, :]
            x = nn.Conv(features=self.features, kernel_size=(3,), strides=(1,), padding="same")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            skip = x
            x = nn.Conv(self.features, kernel_size=(4,), strides=(2,), padding="same")(x)
            return x, skip

    class UpsampleBlock(nn.Module):
        """1D residual upsampling block with skip connection."""

        features: int
        use_time_emb: bool = True

        @nn.compact
        def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, t_emb: Optional[jnp.ndarray]) -> jnp.ndarray:
            x = nn.ConvTranspose(self.features, kernel_size=(4,), strides=(2,), padding="SAME")(x)
            x = jnp.concatenate([x, skip], axis=-1)
            x = nn.Conv(features=self.features, kernel_size=(3,), strides=(1,), padding="same")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            if self.use_time_emb and t_emb is not None:
                t_out = nn.Dense(self.features)(t_emb)
                x = x + t_out[:, None, :]
            x = nn.Conv(features=self.features, kernel_size=(3,), strides=(1,), padding="same")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            return x

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, c: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        flag = False
        if x.ndim == 2:
            x = x[..., None]
            flag = True

        output_channels = x.shape[-1]

        if self.use_condition and c is not None:
            if c.ndim == 2:
                c = c[..., None]
            x = jnp.concatenate([x, c], axis=-1)

        if t is None:
            t_emb = None
            use_time_emb = False
        else:
            t_emb = self.TimeEmbeddingMLP(embed_dim=self.time_emb_dim)(t)
            use_time_emb = True

        skips = []
        h = x
        for ch in self.channel_mult:
            h, skip = self.DownsampleBlock(ch, use_time_emb=use_time_emb)(h, t_emb)
            skips.append(skip)

        h = nn.Conv(features=self.channel_mult[-1], kernel_size=(3,), padding="same")(h)
        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        if use_time_emb:
            t_out = nn.Dense(self.channel_mult[-1])(t_emb)
            h = h + t_out[:, None, :]
        h = nn.Conv(features=self.channel_mult[-1], kernel_size=(3,), padding="same")(h)

        for ch in reversed(self.channel_mult):
            skip = skips.pop()
            h = self.UpsampleBlock(ch, use_time_emb=use_time_emb)(h, skip, t_emb)

        h = nn.Conv(features=output_channels, kernel_size=(1,), padding="same")(h)
        if output_channels == 1 and flag:
            return h[..., 0]
        return h


class UNet2D(nn.Module):
    """2D U-Net with optional time and condition inputs."""

    channel_mult: Tuple[int, ...] = (64, 128, 256)
    time_emb_dim: int = 256
    use_condition: bool = False

    class TimeEmbeddingMLP(nn.Module):
        """Project sinusoidal time features."""

        embed_dim: int
        hidden_dim: int = 128

        @nn.compact
        def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
            x = sinusoidal_time_embedding(t, self.embed_dim)
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.relu(x)
            x = nn.Dense(self.embed_dim)(x)
            return x

    class DownsampleBlock2D(nn.Module):
        """2D residual downsampling block."""

        features: int
        use_time_emb: bool = True

        @nn.compact
        def __call__(self, x: jnp.ndarray, t_emb: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            if self.use_time_emb and t_emb is not None:
                t_out = nn.Dense(self.features)(t_emb)[:, None, None, :]
                x = x + t_out
            x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            skip = x
            x_down = nn.Conv(features=self.features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
            return x_down, skip

    class UpsampleBlock2D(nn.Module):
        """2D residual upsampling block with skip connection."""

        features: int
        use_time_emb: bool = True

        @nn.compact
        def __call__(self, x: jnp.ndarray, skip: jnp.ndarray, t_emb: Optional[jnp.ndarray]) -> jnp.ndarray:
            x = nn.ConvTranspose(features=self.features, kernel_size=(4, 4), strides=(2, 2), padding="SAME")(x)
            x = jnp.concatenate([x, skip], axis=-1)
            x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            if self.use_time_emb and t_emb is not None:
                t_out = nn.Dense(self.features)(t_emb)[:, None, None, :]
                x = x + t_out
            x = nn.Conv(features=self.features, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.GroupNorm()(x)
            x = nn.swish(x)
            return x

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, c: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        flag = False
        if x.ndim == 3:
            flag = True
            x = x[..., None]

        output_channels = x.shape[-1]

        if self.use_condition and c is not None:
            if c.ndim == 3:
                c = c[..., None]
            x = jnp.concatenate([x, c], axis=-1)

        if t is None:
            use_time_emb = False
            t_emb = None
        else:
            use_time_emb = True
            t_emb = self.TimeEmbeddingMLP(embed_dim=self.time_emb_dim)(t)

        skips = []
        h = x
        for ch in self.channel_mult:
            h, skip = self.DownsampleBlock2D(ch, use_time_emb=use_time_emb)(h, t_emb)
            skips.append(skip)

        h = nn.Conv(features=self.channel_mult[-1], kernel_size=(3, 3), strides=(1, 1), padding="SAME")(h)
        h = nn.GroupNorm()(h)
        h = nn.swish(h)
        if use_time_emb:
            t_out = nn.Dense(self.channel_mult[-1])(t_emb)[:, None, None, :]
            h = h + t_out
        h = nn.Conv(features=self.channel_mult[-1], kernel_size=(3, 3), strides=(1, 1), padding="SAME")(h)
        h = nn.GroupNorm()(h)
        h = nn.swish(h)

        for ch in reversed(self.channel_mult):
            skip = skips.pop()
            h = self.UpsampleBlock2D(ch, use_time_emb=use_time_emb)(h, skip, t_emb)

        h = nn.Conv(features=output_channels, kernel_size=(1, 1), strides=(1, 1), padding="SAME")(h)
        if output_channels == 1 and flag:
            return h[..., 0]
        return h


__all__ = [
    "sinusoidal_time_embedding",
    "UNet1D",
    "UNet2D",
]
