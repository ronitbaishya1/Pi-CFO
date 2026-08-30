"""FNO family models for CFO experiments."""

from __future__ import annotations

from typing import Optional

import jax.nn as jnn
import jax.numpy as jnp
from flax import linen as nn


def _sinusoidal_time_embedding(t: jnp.ndarray, embed_dim: int) -> jnp.ndarray:
    """Create sinusoidal time embeddings for FNO conditioning."""
    half_dim = embed_dim // 2
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10000.0), half_dim))
    angles = t[:, None] * freqs[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def compl_mul1d(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Complex multiplication in Fourier space for 1D tensors."""
    return jnp.einsum("bix,iox->box", a, b)


def compl_mul2d(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Complex multiplication in Fourier space for 2D tensors."""
    return jnp.einsum("bixy,ioxy->boxy", a, b)


class SpectralConv1d(nn.Module):
    """1D spectral convolution over truncated Fourier modes."""

    in_channels: int
    out_channels: int
    modes: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.transpose(x, (0, 2, 1))
        scale = 1.0 / (self.in_channels * self.out_channels)
        wr = self.param("wr", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes))
        wi = self.param("wi", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes))
        weights = wr + 1j * wi

        x_ft = jnp.fft.rfft(x, axis=-1)
        out_ft = jnp.zeros((x.shape[0], self.out_channels, x_ft.shape[-1]), dtype=jnp.complex64)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = out_ft.at[:, :, :m].set(compl_mul1d(x_ft[:, :, :m], weights[:, :, :m]))
        x_out = jnp.fft.irfft(out_ft, n=x.shape[-1], axis=-1)
        return jnp.transpose(x_out, (0, 2, 1))


class FNO1d(nn.Module):
    """1D Fourier Neural Operator backbone."""

    num_channels: int
    modes: int = 16
    width: int = 64
    num_blocks: int = 4
    use_condition: bool = False
    use_time: bool = True

    def setup(self):
        self.fc0 = nn.Dense(self.width)
        self.convs = [SpectralConv1d(self.width, self.width, self.modes) for _ in range(self.num_blocks)]
        self.ws = [nn.Conv(self.width, kernel_size=(1,)) for _ in range(self.num_blocks)]
        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(self.num_channels)

    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, c: Optional[jnp.ndarray] = None, grid: Optional[jnp.ndarray] = None):
        if x.ndim == 2:
            x = x[..., None]
        B, S, _ = x.shape
        if grid is None:
            g = jnp.linspace(0.0, 1.0, S)
            grid = jnp.tile(g[None, :, None], (B, 1, 1))
        x = jnp.concatenate([x, grid], axis=-1)
        if self.use_condition and c is not None:
            if c.ndim == 2:
                c = c[..., None]
            x = jnp.concatenate([x, c], axis=-1)

        x = self.fc0(x)
        if self.use_time and t is not None:
            x = x + _sinusoidal_time_embedding(t, self.width)[:, None, :]

        for conv, w in zip(self.convs[:-1], self.ws[:-1]):
            x = jnn.gelu(conv(x) + w(x))
        x = self.convs[-1](x) + self.ws[-1](x)
        x = jnn.gelu(self.fc1(x))
        return self.fc2(x)


class SpectralConv2d(nn.Module):
    """2D spectral convolution over truncated Fourier modes."""

    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.transpose(x, (0, 3, 1, 2))
        scale = 1.0 / (self.in_channels * self.out_channels)
        wr1 = self.param("wr1", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes1, self.modes2))
        wi1 = self.param("wi1", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes1, self.modes2))
        wr2 = self.param("wr2", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes1, self.modes2))
        wi2 = self.param("wi2", nn.initializers.normal(stddev=scale), (self.in_channels, self.out_channels, self.modes1, self.modes2))
        w1 = wr1 + 1j * wi1
        w2 = wr2 + 1j * wi2

        x_ft = jnp.fft.rfft2(x, axes=(-2, -1))
        out_ft = jnp.zeros((x.shape[0], self.out_channels, x.shape[-2], x_ft.shape[-1]), dtype=jnp.complex64)
        m1 = min(self.modes1, x.shape[-2])
        m2 = min(self.modes2, x_ft.shape[-1])
        out_ft = out_ft.at[:, :, :m1, :m2].set(compl_mul2d(x_ft[:, :, :m1, :m2], w1[:, :, :m1, :m2]))
        out_ft = out_ft.at[:, :, -m1:, :m2].set(compl_mul2d(x_ft[:, :, -m1:, :m2], w2[:, :, :m1, :m2]))
        x_out = jnp.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]), axes=(-2, -1))
        return jnp.transpose(x_out, (0, 2, 3, 1))


class FNO2d(nn.Module):
    """2D Fourier Neural Operator backbone."""

    num_channels: int
    modes1: int = 12
    modes2: int = 12
    width: int = 64
    num_blocks: int = 4
    use_condition: bool = False
    use_time: bool = False

    def setup(self):
        self.fc0 = nn.Dense(self.width)
        self.convs = [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_blocks)]
        self.ws = [nn.Conv(self.width, kernel_size=(1, 1)) for _ in range(self.num_blocks)]
        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(self.num_channels)

    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, c: Optional[jnp.ndarray] = None, grid: Optional[jnp.ndarray] = None):
        if x.ndim == 3:
            x = x[..., None]
        N, H, W, _ = x.shape

        if grid is None:
            gy = jnp.linspace(0.0, 1.0, H)
            gx = jnp.linspace(0.0, 1.0, W)
            yy, xx = jnp.meshgrid(gy, gx, indexing="ij")
            grid = jnp.stack([yy, xx], axis=-1)
            grid = jnp.tile(grid[None, ...], (N, 1, 1, 1))

        x = jnp.concatenate([x, grid], axis=-1)
        if self.use_condition and c is not None:
            if c.ndim == 3:
                c = c[..., None]
            x = jnp.concatenate([x, c], axis=-1)

        x = self.fc0(x)
        if self.use_time and t is not None:
            x = x + _sinusoidal_time_embedding(t, self.width)[:, None, None, :]

        for conv, w in zip(self.convs[:-1], self.ws[:-1]):
            x = jnn.gelu(conv(x) + w(x))
        x = self.convs[-1](x) + self.ws[-1](x)
        x = jnn.gelu(self.fc1(x))
        return self.fc2(x)


__all__ = [
    "SpectralConv1d",
    "SpectralConv2d",
    "FNO1d",
    "FNO2d",
]
