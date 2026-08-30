"""Simple MLP baseline models."""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn


def time_encoding(t_unit: jnp.ndarray, L: int = 8):
    """Build sinusoidal time features from normalized scalar time inputs."""
    t = jnp.asarray(t_unit)
    if t.ndim == 0:
        t = t[None]
    if t.ndim == 1:
        t = t[:, None]
    freqs = 2.0 ** jnp.arange(L)
    phases = 2.0 * jnp.pi * t * freqs[None, :]
    return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)], axis=-1)


class SimpleMLP(nn.Module):
    """Residual MLP backbone with optional time and condition inputs."""

    hidden_dims: tuple = (128, 256, 256, 256, 128)
    time_embed_L: int = 16
    residual: bool = True
    use_condition: bool = False
    output_dim: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: Optional[jnp.ndarray] = None, c: Optional[jnp.ndarray] = None):
        """Predict velocity/next-state for each batch element."""
        bsz = x.shape[0]

        if t is not None:
            if self.time_embed_L > 0:
                te = time_encoding(t, L=self.time_embed_L)
                if te.shape[0] != bsz:
                    if te.shape[0] == 1:
                        te = jnp.broadcast_to(te, (bsz, te.shape[-1]))
                    else:
                        raise ValueError(f"Batch mismatch: x has B={bsz}, but time features have {te.shape[0]}")
            else:
                t_arr = jnp.asarray(t)
                if t_arr.ndim == 0:
                    t_arr = jnp.repeat(t_arr[None], repeats=bsz, axis=0)[:, None]
                elif t_arr.ndim == 1:
                    t_arr = t_arr[:, None]
                te = t_arr
        else:
            te = jnp.zeros((bsz, 2 * self.time_embed_L if self.time_embed_L > 0 else 1))

        h = jnp.concatenate([x, te], axis=-1)
        if self.use_condition and c is not None:
            h = jnp.concatenate([h, c], axis=-1)

        for width in self.hidden_dims:
            h = nn.relu(nn.Dense(width)(h))
        out = nn.Dense(self.output_dim)(h)
        return x + out if self.residual else out


__all__ = ["time_encoding", "SimpleMLP"]
