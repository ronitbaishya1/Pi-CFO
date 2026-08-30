from __future__ import annotations

"""Numerical integration helpers for CFO velocity fields."""

from typing import Any

import jax
import jax.numpy as jnp


@jax.jit
def euler_step(
    state: Any,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    condition: jnp.ndarray | None,
    delta_t: float | jnp.ndarray,
) -> jnp.ndarray:
    """Advance one explicit Euler step."""
    velocity = state.apply_fn({'params': state.params}, x0, t, condition)
    return x0 + delta_t * velocity


@jax.jit
def rk4_step(
    state: Any,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    condition: jnp.ndarray | None,
    delta_t: float | jnp.ndarray,
) -> jnp.ndarray:
    """Advance one classical Runge-Kutta (RK4) step."""
    k1 = state.apply_fn({'params': state.params}, x0, t, condition)
    k2 = state.apply_fn({'params': state.params}, x0 + 0.5 * delta_t * k1, t + 0.5 * delta_t, condition)
    k3 = state.apply_fn({'params': state.params}, x0 + 0.5 * delta_t * k2, t + 0.5 * delta_t, condition)
    k4 = state.apply_fn({'params': state.params}, x0 + delta_t * k3, t + delta_t, condition)
    return x0 + (delta_t / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def heun_step(
    state: Any,
    x0: jnp.ndarray,
    t: jnp.ndarray,
    condition: jnp.ndarray | None,
    delta_t: float | jnp.ndarray,
) -> jnp.ndarray:
    """Advance one Heun (improved Euler) step."""
    k1 = state.apply_fn({'params': state.params}, x0, t, condition)
    x_predictor = x0 + delta_t * k1
    k2 = state.apply_fn({'params': state.params}, x_predictor, t + delta_t, condition)
    return x0 + 0.5 * delta_t * (k1 + k2)


INTEGRATOR_STEP_FNS = {
    "Euler": euler_step,
    "RK4": rk4_step,
    "Heun": heun_step,
}
