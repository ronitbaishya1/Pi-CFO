"""Spline and derivative utilities used by CFO training."""

from __future__ import annotations

import gc
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def linear_spline(X, time=None):
    """Compute vectorized piecewise-linear spline coefficients."""
    X = np.asarray(X)
    if X.ndim < 2:
        raise ValueError("Input X must have shape (B, T, ...), with T >= 2.")
    B, T = X.shape[0], X.shape[1]
    if T < 2:
        raise ValueError("Need at least two time points (T >= 2) for piecewise linear interpolation.")

    a0 = X[:, :-1, ...]
    a1 = X[:, 1:, ...] - X[:, :-1, ...]
    coefs = np.stack((a0, a1), axis=2).reshape(-1, 2, *X.shape[2:])

    if time is None:
        k = np.arange(T - 1, dtype=np.int32)
        start_ind = np.broadcast_to(k, (B, T - 1)).reshape(-1)
        return coefs, start_ind

    time = np.asarray(time, dtype=X.dtype)
    if time.shape != (B, T):
        raise ValueError(f"Shape of time must be ({B}, {T}), but got {time.shape}")
    if np.any(np.diff(time, axis=1) <= 0):
        raise ValueError("Time must be strictly monotonically increasing along the T dimension.")

    return coefs, time[:, :-1].reshape(-1), time[:, 1:].reshape(-1)


def estimate_derivatives(x: jnp.ndarray, time=None):
    """Estimate first and second derivatives at spline knots."""
    T = x.shape[0]
    if T < 3:
        return jnp.gradient(x, axis=0), jnp.gradient(jnp.gradient(x, axis=0), axis=0)

    if time is None:
        dt = 1.0 / (T - 1)
        dx_left = (-3.0 * x[0] + 4.0 * x[1] - 1.0 * x[2]) / (2.0 * dt)
        dx_right = (3.0 * x[-1] - 4.0 * x[-2] + 1.0 * x[-3]) / (2.0 * dt)
        dx_c = (x[2:] - x[:-2]) / (2.0 * dt)
        dx = jnp.concatenate([dx_left[None], dx_c, dx_right[None]], axis=0)

        c2 = (x[2:] - 2 * x[1:-1] + x[:-2]) / (dt**2)
        ddx = jnp.concatenate([c2[0:1], c2, c2[-1:]], axis=0)
        return dx, ddx

    h = jnp.diff(time)
    h = h.reshape(h.shape + (1,) * (x.ndim - h.ndim))
    h0, h1 = h[:-1], h[1:]

    w_im1 = -h1 / (h0 * (h0 + h1))
    w_i = (h1 - h0) / (h0 * h1)
    w_ip1 = h0 / (h1 * (h0 + h1))
    dx_c = w_im1 * x[:-2] + w_i * x[1:-1] + w_ip1 * x[2:]

    h0L, h1L = h[0], h[1]
    dx_left = (-(2.0 * h0L + h1L) / (h0L * (h0L + h1L))) * x[0] + ((h0L + h1L) / (h0L * h1L)) * x[1] - (h0L / (h1L * (h0L + h1L))) * x[2]
    g0, g1 = h[-2], h[-1]
    dx_right = (g1 / (g0 * (g0 + g1))) * x[-3] - ((g0 + g1) / (g0 * g1)) * x[-2] + ((2.0 * g1 + g0) / (g1 * (g0 + g1))) * x[-1]
    dx = jnp.concatenate([dx_left[None], dx_c, dx_right[None]], axis=0)

    ddx_c = 2.0 * (((x[2:] - x[1:-1]) / h1) - ((x[1:-1] - x[:-2]) / h0)) / (h0 + h1)
    ddx_left = 2.0 * (x[0] / (h0L * (h0L + h1L)) - x[1] / (h0L * h1L) + x[2] / (h1L * (h0L + h1L)))
    ddx_right = 2.0 * (x[-3] / (g0 * (g0 + g1)) - x[-2] / (g0 * g1) + x[-1] / (g1 * (g0 + g1)))
    ddx = jnp.concatenate([ddx_left[None], ddx_c, ddx_right[None]], axis=0)
    return dx, ddx


def estimate_derivatives_high(x: jnp.ndarray, time=None):
    """Higher-order derivative estimate for uniform-time data."""
    if time is not None:
        return estimate_derivatives(x, time=time)
    return estimate_derivatives(x, time=None)


def eval_spline_at_times(coefs, start_time, end_time, t):
    """Evaluate piecewise quintic (or generic τ-polynomial) at query times."""
    coefs = np.asarray(coefs)
    st = np.asarray(start_time)
    en = np.asarray(end_time)
    t = np.asarray(t, dtype=np.float64)

    idx = np.searchsorted(st, t, side="right") - 1
    idx = np.clip(idx, 0, len(st) - 1)
    idx = np.where(t > en[-1], len(st) - 1, idx)

    tau = (t - st[idx]) / (en[idx] - st[idx])
    tau = np.clip(tau, 0.0, 1.0)

    a0 = coefs[idx, 0]
    a1 = coefs[idx, 1]
    a2 = coefs[idx, 2]
    a3 = coefs[idx, 3]
    a4 = coefs[idx, 4]
    a5 = coefs[idx, 5]
    y = (((((a5 * tau[..., None] + a4) * tau[..., None] + a3) * tau[..., None] + a2) * tau[..., None] + a1) * tau[..., None] + a0)
    return y


def quintic_spline_onesample(x: jnp.ndarray, time=None, num_point=3):
    """Compute quintic spline coefficients for one trajectory."""
    T = x.shape[0]
    if T < 2:
        raise ValueError("Need at least two points")

    dx, ddx = estimate_derivatives(x, time) if num_point == 3 else estimate_derivatives_high(x, time)
    x0, x1 = x[:-1], x[1:]
    dx0, dx1 = dx[:-1], dx[1:]
    ddx0, ddx1 = ddx[:-1], ddx[1:]

    if time is None:
        dt = 1.0 / (T - 1)
        a0 = x0
        a1 = dx0 * dt
        a2 = 0.5 * ddx0 * (dt**2)
        S1 = x1 - (a0 + a1 + a2)
        S2 = dx1 * dt - (a1 + 2 * a2)
        S3 = ddx1 * (dt**2) - (2 * a2)
        B_vec = jnp.stack([S1, S2, S3], axis=-1)
        M_inv = jnp.array([[10.0, -4.0, 0.5], [-15.0, 7.0, -1.0], [6.0, -3.0, 0.5]], dtype=x.dtype)
        a345 = B_vec @ M_inv.T
        coeffs = jnp.stack([a0, a1, a2, a345[..., 0], a345[..., 1], a345[..., 2]], axis=1).reshape(-1, 6, *x.shape[1:])
        start_ind = jnp.arange(T - 1, dtype=jnp.int32)
        return coeffs, start_ind

    dt = jnp.diff(time).reshape(T - 1, *(1,) * (x.ndim - 1))
    a0 = x0
    a1 = dx0 * dt
    a2 = 0.5 * ddx0 * (dt**2)
    S1 = x1 - (a0 + a1 + a2)
    S2 = dx1 * dt - (a1 + 2 * a2)
    S3 = ddx1 * (dt**2) - (2 * a2)
    B_vec = jnp.stack([S1, S2, S3], axis=-1)
    M_inv = jnp.array([[10.0, -4.0, 0.5], [-15.0, 7.0, -1.0], [6.0, -3.0, 0.5]], dtype=x.dtype)
    a345 = B_vec @ M_inv.T
    coeffs = jnp.stack([a0, a1, a2, a345[..., 0], a345[..., 1], a345[..., 2]], axis=1)
    return coeffs, time[:-1], time[1:]


@partial(jax.jit, static_argnames=("num_point",))
def quintic_spline(X, time=None, num_point=3):
    """Compute quintic spline coefficients for a batch of samples."""
    spline_fn = partial(quintic_spline_onesample, num_point=num_point)
    if time is None:
        return jax.vmap(spline_fn, in_axes=0)(X)
    return jax.vmap(spline_fn, in_axes=(0, 0))(X, time)


def quintic_spline_batch(X, time=None, batch_size=64, num_point=3):
    """NumPy wrapper for batched quintic spline computation."""
    num_samples = len(X)
    if time is None:
        all_coeffs, all_inds = [], []
        for i in range(0, num_samples, batch_size):
            batch_coeffs, batch_ind = quintic_spline(jnp.array(X[i : i + batch_size]), num_point=num_point)
            all_coeffs.append(np.array(batch_coeffs).reshape(-1, *batch_coeffs.shape[2:]))
            all_inds.append(np.array(batch_ind).reshape(-1))
            gc.collect()
        return np.concatenate(all_coeffs, axis=0), np.concatenate(all_inds, axis=0)

    all_coeffs, all_starts, all_ends = [], [], []
    for i in range(0, num_samples, batch_size):
        xb = jnp.array(X[i : i + batch_size])
        tb = jnp.array(time[i : i + batch_size])
        batch_coeffs, batch_starts, batch_ends = quintic_spline(xb, time=tb, num_point=num_point)
        all_coeffs.append(np.array(batch_coeffs).reshape(-1, *batch_coeffs.shape[2:]))
        all_starts.append(np.array(batch_starts).reshape(-1))
        all_ends.append(np.array(batch_ends).reshape(-1))
        gc.collect()
    return np.concatenate(all_coeffs, axis=0), np.concatenate(all_starts, axis=0), np.concatenate(all_ends, axis=0)


__all__ = [
    "linear_spline",
    "estimate_derivatives",
    "estimate_derivatives_high",
    "eval_spline_at_times",
    "quintic_spline_onesample",
    "quintic_spline",
    "quintic_spline_batch",
]
