from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.experimental.ode import odeint

from utils.integrators import INTEGRATOR_STEP_FNS


class ContinuousFlowOperator:
    """Algorithm-only CFO definition.

    This class contains only CFO math/algorithm logic and stateless inference calls.
    State creation, optimization, training loops, and checkpointing are handled outside.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        input_shape: Sequence[int],
        gamma: float = 1e-5,
        spline_type: str = "quintic",
        use_condition: bool = False,
        condition_shape: Sequence[int] | None = None,
    ):
        self.model = model
        self.input_shape = tuple(input_shape)
        self.gamma = float(gamma)
        self.use_condition = bool(use_condition)
        self.condition_shape = tuple(condition_shape) if condition_shape is not None else None
        self.spline_type = str(spline_type)
        if self.spline_type not in {"linear", "quintic"}:
            raise ValueError("`spline_type` must be one of {'linear', 'quintic'}.")
        if self.use_condition and self.condition_shape is None:
            raise ValueError("`condition_shape` must be provided when `use_condition=True`.")

    @staticmethod
    def _reshape_time_like(x: jnp.ndarray, spline_coef: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(x, (x.shape[0],) + (1,) * (spline_coef.ndim - 2))

    def _model_apply(self, params, x, t, condition=None):
        return self.model.apply({'params': params}, x, t, condition)

    def sample_conditional_path(self, tau, spline_coef, eps, dt=None):
        if dt is None:
            raise ValueError("`dt` must be provided to sample the conditional path.")

        if self.spline_type == "linear":
            mu_t = spline_coef[:, 0] + tau * spline_coef[:, 1]
        else:
            mu_t = (
                spline_coef[:, 0]
                + tau * spline_coef[:, 1]
                + tau**2 * spline_coef[:, 2]
                + tau**3 * spline_coef[:, 3]
                + tau**4 * spline_coef[:, 4]
                + tau**5 * spline_coef[:, 5]
            )

        gamma_t = self.gamma * (tau**3) * ((1.0 - tau) ** 3)
        return mu_t + gamma_t * eps

    def compute_targets(self, spline_coef: jnp.ndarray, tau: jnp.ndarray, dt: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
        gamma_prime = self.gamma * (3.0 / dt) * (tau**2 * (1.0 - tau) ** 2 * (1.0 - 2.0 * tau))
        if self.spline_type == "linear":
            return (1 / dt) * spline_coef[:, 1] + gamma_prime * eps
        return (1 / dt) * (
            spline_coef[:, 1]
            + 2 * spline_coef[:, 2] * tau
            + 3 * spline_coef[:, 3] * tau**2
            + 4 * spline_coef[:, 4] * tau**3
            + 5 * spline_coef[:, 5] * tau**4
        ) + gamma_prime * eps

    def loss_fn(self, params, batch):
        if self.use_condition:
            spline_coef, condition, t_start, t_end, delta_t, eps = batch
        else:
            spline_coef, t_start, t_end, delta_t, eps = batch
            condition = None

        dt = t_end - t_start
        tau = delta_t / dt
        tau = self._reshape_time_like(tau, spline_coef)
        dt = self._reshape_time_like(dt, spline_coef)

        x = self.sample_conditional_path(tau, spline_coef, eps, dt)
        outputs = self._model_apply(params, x, t_start + delta_t, condition)
        targets = self.compute_targets(spline_coef, tau, dt, eps)
        return jnp.mean((outputs - targets) ** 2)

    def infer_at(self, state, x_at_s, s, t, steps=50, condition=None, method="RK4"):
        if method not in INTEGRATOR_STEP_FNS:
            raise ValueError("method must be 'Euler', 'Heun', or 'RK4'.")

        step_fn = INTEGRATOR_STEP_FNS[method]
        steps = max(int(steps), 2)
        delta_t = (t - s) / (steps - 1)
        t_values = jnp.linspace(s, t, steps)

        x = x_at_s
        for i in range(steps - 1):
            t_batch = jnp.full((x.shape[0],), t_values[i])
            x = step_fn(state, x, t_batch, condition, delta_t)
        return x
    
    def uniform_inference(self, state, x_0, trajectory_points_num: int, steps_per_segment=3, condition=None, method="RK4"):
        points_num = int(trajectory_points_num)
        segment_steps = max(int(steps_per_segment), 1)
        segment_dt = 1.0 / max(points_num - 1, 1)

        preds = [np.array(x_0)]
        x = x_0
        for idx in range(points_num - 1):
            s = idx * segment_dt
            t = (idx + 1) * segment_dt
            x = self.infer_at(
                state,
                x,
                s=s,
                t=t,
                steps=segment_steps + 1,
                condition=condition,
                method=method,
            )
            preds.append(np.array(x))

        pred = np.stack(preds, axis=0)
        return np.swapaxes(pred, 0, 1)

    
