from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from utils.integrators import INTEGRATOR_STEP_FNS


class ContinuousFlowOperator:
    """Continuous Flow Operator (CFO).

    This class contains the mathematical CFO algorithm and stateless
    inference operations.

    Training loops, optimizer construction, checkpoints, and TrainState
    management are handled elsewhere.

    The clean spline state s(t) is exposed separately through
    ``spline_mean()`` so that later physics-informed extensions can
    impose PDE constraints on the physical trajectory without applying
    those constraints to the stochastic CFO perturbation.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

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

        self.input_shape = tuple(
            input_shape
        )

        self.gamma = float(
            gamma
        )

        self.spline_type = str(
            spline_type
        )

        self.use_condition = bool(
            use_condition
        )

        self.condition_shape = (
            tuple(condition_shape)
            if condition_shape is not None
            else None
        )

        # --------------------------------------------------------
        # Validate spline type
        # --------------------------------------------------------

        if self.spline_type not in {
            "linear",
            "quintic",
        }:
            raise ValueError(
                "`spline_type` must be one of "
                "{'linear', 'quintic'}."
            )

        # --------------------------------------------------------
        # Validate conditioning setup
        # --------------------------------------------------------

        if (
            self.use_condition
            and self.condition_shape is None
        ):
            raise ValueError(
                "`condition_shape` must be provided "
                "when `use_condition=True`."
            )

    # ============================================================
    # TIME RESHAPING
    # ============================================================

    @staticmethod
    def _reshape_time_like(
        x: jnp.ndarray,
        spline_coef: jnp.ndarray,
    ) -> jnp.ndarray:
        """Reshape time values so they broadcast over spatial fields.

        Example
        -------

        If:

            x.shape = (B,)

        and:

            spline_coef.shape = (B, 6, Nx, Ny, C)

        then this returns:

            (B, 1, 1, 1)

        so that time can broadcast against arrays of shape:

            (B, Nx, Ny, C)
        """

        return jnp.reshape(
            x,
            (
                x.shape[0],
            )
            + (1,)
            * (
                spline_coef.ndim - 2
            ),
        )

    # ============================================================
    # MODEL FORWARD PASS
    # ============================================================

    def _model_apply(
        self,
        params,
        x,
        t,
        condition=None,
    ):
        """Apply the neural operator.

        Parameters
        ----------
        params
            Neural-network parameters.

        x
            Current physical/CFO state.

        t
            Physical time.

        condition
            Optional conditioning field, later useful for
            bathymetry b(x, y).
        """

        return self.model.apply(
            {
                "params": params,
            },
            x,
            t,
            condition,
        )

    # ============================================================
    # CLEAN SPLINE STATE s(t)
    # ============================================================

    def spline_mean(
        self,
        tau: jnp.ndarray,
        spline_coef: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate the clean temporal spline state s(t).

        IMPORTANT
        ---------
        No CFO stochastic perturbation is applied here.

        This is the state that will later be used by PI-CFO for
        evaluating the shallow-water PDE residual.

        CFO flow matching will continue to use:

            I(t) = s(t) + gamma(t) * epsilon
        """

        # --------------------------------------------------------
        # Linear spline
        # --------------------------------------------------------

        if self.spline_type == "linear":

            return (
                spline_coef[:, 0]
                +
                tau
                * spline_coef[:, 1]
            )

        # --------------------------------------------------------
        # Quintic Hermite spline
        #
        # s(tau)
        #
        # = a0
        # + a1 tau
        # + a2 tau^2
        # + a3 tau^3
        # + a4 tau^4
        # + a5 tau^5
        # --------------------------------------------------------

        return (
            spline_coef[:, 0]
            +
            tau
            * spline_coef[:, 1]
            +
            tau**2
            * spline_coef[:, 2]
            +
            tau**3
            * spline_coef[:, 3]
            +
            tau**4
            * spline_coef[:, 4]
            +
            tau**5
            * spline_coef[:, 5]
        )

    # ============================================================
    # CFO STOCHASTIC CONDITIONAL PATH I(t)
    # ============================================================

    def sample_conditional_path(
        self,
        tau: jnp.ndarray,
        spline_coef: jnp.ndarray,
        eps: jnp.ndarray,
        dt: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Construct the CFO conditional path.

        CFO uses:

            I(t)
            =
            s(t)
            +
            gamma(tau) * epsilon

        where:

            gamma(tau)
            =
            gamma * tau^3 * (1 - tau)^3

        Therefore the stochastic perturbation vanishes at:

            tau = 0

        and:

            tau = 1.
        """

        if dt is None:
            raise ValueError(
                "`dt` must be provided to sample "
                "the conditional path."
            )

        # --------------------------------------------------------
        # 1. Clean physical spline
        # --------------------------------------------------------

        mu_t = self.spline_mean(
            tau,
            spline_coef,
        )

        # --------------------------------------------------------
        # 2. CFO stochastic perturbation
        # --------------------------------------------------------

        gamma_t = (
            self.gamma
            *
            tau**3
            *
            (1.0 - tau) ** 3
        )

        # --------------------------------------------------------
        # 3. CFO conditional state
        #
        # I(t) = s(t) + gamma(t) eps
        # --------------------------------------------------------

        x = (
            mu_t
            +
            gamma_t
            * eps
        )

        return x

    # ============================================================
    # CFO ANALYTIC VELOCITY TARGET
    # ============================================================

    def compute_targets(
        self,
        spline_coef: jnp.ndarray,
        tau: jnp.ndarray,
        dt: jnp.ndarray,
        eps: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the analytic CFO velocity target dI/dt."""

        # --------------------------------------------------------
        # Derivative of stochastic perturbation
        #
        # gamma(tau)
        # =
        # gamma tau^3 (1-tau)^3
        #
        # d gamma / dt
        # =
        # (3 gamma / dt)
        # tau^2 (1-tau)^2 (1-2tau)
        # --------------------------------------------------------

        gamma_prime = (
            self.gamma
            *
            (3.0 / dt)
            *
            (
                tau**2
                *
                (1.0 - tau) ** 2
                *
                (1.0 - 2.0 * tau)
            )
        )

        # --------------------------------------------------------
        # Linear spline target
        # --------------------------------------------------------

        if self.spline_type == "linear":

            spline_velocity = (
                (1.0 / dt)
                *
                spline_coef[:, 1]
            )

            stochastic_velocity = (
                gamma_prime
                *
                eps
            )

            return (
                spline_velocity
                +
                stochastic_velocity
            )

        # --------------------------------------------------------
        # Quintic spline derivative
        # --------------------------------------------------------

        spline_velocity = (
            (1.0 / dt)
            *
            (
                spline_coef[:, 1]
                +
                2.0
                * spline_coef[:, 2]
                * tau
                +
                3.0
                * spline_coef[:, 3]
                * tau**2
                +
                4.0
                * spline_coef[:, 4]
                * tau**3
                +
                5.0
                * spline_coef[:, 5]
                * tau**4
            )
        )

        stochastic_velocity = (
            gamma_prime
            *
            eps
        )

        return (
            spline_velocity
            +
            stochastic_velocity
        )

    # ============================================================
    # ORIGINAL CFO TRAINING LOSS
    # ============================================================

    def loss_fn(
        self,
        params,
        batch,
    ):
        """Compute the original CFO flow-matching loss.

        NOTE
        ----
        This remains the original CFO objective.

        No SWE physics term is added here.

        Later PI-CFO will subclass this class and introduce:

            L_total
            =
            L_CFO
            +
            lambda_PDE L_PDE.
        """

        # --------------------------------------------------------
        # 1. Unpack batch
        # --------------------------------------------------------

        if self.use_condition:

            (
                spline_coef,
                condition,
                t_start,
                t_end,
                delta_t,
                eps,
            ) = batch

        else:

            (
                spline_coef,
                t_start,
                t_end,
                delta_t,
                eps,
            ) = batch

            condition = None

        # --------------------------------------------------------
        # 2. Physical duration of spline segment
        # --------------------------------------------------------

        dt = (
            t_end
            -
            t_start
        )

        # --------------------------------------------------------
        # 3. Convert physical local time to normalized tau
        #
        # tau in [0, 1]
        # --------------------------------------------------------

        tau = (
            delta_t
            /
            dt
        )

        # --------------------------------------------------------
        # 4. Reshape tau and dt for spatial broadcasting
        # --------------------------------------------------------

        tau = self._reshape_time_like(
            tau,
            spline_coef,
        )

        dt_reshaped = self._reshape_time_like(
            dt,
            spline_coef,
        )

        # --------------------------------------------------------
        # 5. Construct stochastic CFO state
        #
        # I(t) = s(t) + gamma(t) eps
        # --------------------------------------------------------

        x = self.sample_conditional_path(
            tau,
            spline_coef,
            eps,
            dt_reshaped,
        )

        # --------------------------------------------------------
        # 6. Physical time supplied to network
        # --------------------------------------------------------

        physical_time = (
            t_start
            +
            delta_t
        )

        # --------------------------------------------------------
        # 7. Neural operator prediction
        #
        # N_theta(t, I(t))
        # --------------------------------------------------------

        outputs = self._model_apply(
            params,
            x,
            physical_time,
            condition,
        )

        # --------------------------------------------------------
        # 8. Analytic CFO velocity target
        #
        # dI/dt
        # --------------------------------------------------------

        targets = self.compute_targets(
            spline_coef,
            tau,
            dt_reshaped,
            eps,
        )

        # --------------------------------------------------------
        # 9. CFO flow-matching loss
        # --------------------------------------------------------

        cfo_loss = jnp.mean(
            (
                outputs
                -
                targets
            ) ** 2
        )

        return cfo_loss

    # ============================================================
    # INFERENCE BETWEEN TWO TIMES
    # ============================================================

    def infer_at(
        self,
        state,
        x_at_s,
        s,
        t,
        steps=50,
        condition=None,
        method="RK4",
    ):
        """Integrate the learned continuous vector field from s to t."""

        # --------------------------------------------------------
        # Validate numerical integrator
        # --------------------------------------------------------

        if method not in INTEGRATOR_STEP_FNS:

            raise ValueError(
                "method must be "
                "'Euler', 'Heun', or 'RK4'."
            )

        step_fn = (
            INTEGRATOR_STEP_FNS[
                method
            ]
        )

        # --------------------------------------------------------
        # Need at least two integration points
        # --------------------------------------------------------

        steps = max(
            int(steps),
            2,
        )

        # --------------------------------------------------------
        # Time step
        # --------------------------------------------------------

        delta_t = (
            (t - s)
            /
            (steps - 1)
        )

        # --------------------------------------------------------
        # Integration time grid
        # --------------------------------------------------------

        t_values = jnp.linspace(
            s,
            t,
            steps,
        )

        x = x_at_s

        # --------------------------------------------------------
        # Integrate learned neural ODE
        # --------------------------------------------------------

        for i in range(
            steps - 1
        ):

            t_batch = jnp.full(
                (
                    x.shape[0],
                ),
                t_values[i],
            )

            x = step_fn(
                state,
                x,
                t_batch,
                condition,
                delta_t,
            )

        return x

    # ============================================================
    # FULL UNIFORM TRAJECTORY INFERENCE
    # ============================================================

    def uniform_inference(
        self,
        state,
        x_0,
        trajectory_points_num: int,
        steps_per_segment=3,
        condition=None,
        method="RK4",
    ):
        """Generate a full CFO trajectory on a uniform time grid."""

        # --------------------------------------------------------
        # Number of desired trajectory snapshots
        # --------------------------------------------------------

        points_num = int(
            trajectory_points_num
        )

        # --------------------------------------------------------
        # Internal ODE integration steps per output segment
        # --------------------------------------------------------

        segment_steps = max(
            int(
                steps_per_segment
            ),
            1,
        )

        # --------------------------------------------------------
        # Normalized output-time spacing
        #
        # CFO trajectories live on t in [0, 1]
        # --------------------------------------------------------

        segment_dt = (
            1.0
            /
            max(
                points_num - 1,
                1,
            )
        )

        # --------------------------------------------------------
        # First prediction is initial condition
        # --------------------------------------------------------

        preds = [
            np.array(
                x_0
            )
        ]

        x = x_0

        # --------------------------------------------------------
        # Roll forward one output segment at a time
        # --------------------------------------------------------

        for idx in range(
            points_num - 1
        ):

            # Segment start
            s = (
                idx
                *
                segment_dt
            )

            # Segment end
            t = (
                (idx + 1)
                *
                segment_dt
            )

            # ----------------------------------------------------
            # Integrate learned continuous vector field
            # ----------------------------------------------------

            x = self.infer_at(
                state,
                x,
                s=s,
                t=t,
                steps=segment_steps + 1,
                condition=condition,
                method=method,
            )

            preds.append(
                np.array(
                    x
                )
            )

        # --------------------------------------------------------
        # Stack:
        #
        # (time, batch, ...)
        # --------------------------------------------------------

        pred = np.stack(
            preds,
            axis=0,
        )

        # --------------------------------------------------------
        # Convert to:
        #
        # (batch, time, ...)
        # --------------------------------------------------------

        pred = np.swapaxes(
            pred,
            0,
            1,
        )

        return pred