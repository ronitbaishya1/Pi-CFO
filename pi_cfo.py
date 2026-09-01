from __future__ import annotations

import jax.numpy as jnp

from cfo import ContinuousFlowOperator
from utils.physics_swe import swe_physics_loss


class PhysicsInformedCFO(ContinuousFlowOperator):
    """Physics-Informed Continuous Flow Operator for 2D SWE.

    The model is trained with two objectives:

    1. Original CFO flow-matching loss

        L_CFO = ||N_theta(t, I(t)) - dI/dt||^2

       where

        I(t) = s(t) + gamma(t) * epsilon

    2. Shallow-water physics loss

        L_PDE = ||q_t + div(F(q))||^2

       where the clean spline state is

        q = s(t)

       and

        q_t = N_theta(t, q).

    The total objective is

        L_total
        =
        L_CFO
        +
        lambda_pde * L_PDE.
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        *,
        model,
        input_shape,
        gamma: float = 1e-5,
        spline_type: str = "quintic",
        use_condition: bool = False,
        condition_shape=None,
        lambda_pde: float = 1e-3,
        dx: float = 0.15625,
        dy: float = 0.15625,
        gravity: float = 1.0,
    ):

        # --------------------------------------------------------
        # Initialize original CFO
        # --------------------------------------------------------

        super().__init__(
            model=model,
            input_shape=input_shape,
            gamma=gamma,
            spline_type=spline_type,
            use_condition=use_condition,
            condition_shape=condition_shape,
        )

        # --------------------------------------------------------
        # Physics-informed parameters
        # --------------------------------------------------------

        self.lambda_pde = float(
            lambda_pde
        )

        self.dx = float(
            dx
        )

        self.dy = float(
            dy
        )

        self.gravity = float(
            gravity
        )

    # ============================================================
    # LOSS COMPONENTS
    # ============================================================

    def loss_components(
        self,
        params,
        batch,
    ):
        """Compute CFO, PDE, and total losses separately.

        Returns
        -------
        total_loss
            L_CFO + lambda_pde * L_PDE

        cfo_loss
            Original CFO flow-matching loss.

        pde_loss
            Shallow-water PDE residual loss.
        """

        # ========================================================
        # 1. UNPACK TRAINING BATCH
        # ========================================================

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

        # ========================================================
        # 2. CURRENT SPLINE SEGMENT DURATION
        # ========================================================

        dt = (
            t_end
            -
            t_start
        )

        # ========================================================
        # 3. LOCAL NORMALIZED SPLINE TIME
        #
        # tau in [0, 1]
        # ========================================================

        tau = (
            delta_t
            /
            dt
        )

        # ========================================================
        # 4. RESHAPE TIME FOR SPATIAL BROADCASTING
        # ========================================================

        tau_reshaped = self._reshape_time_like(
            tau,
            spline_coef,
        )

        dt_reshaped = self._reshape_time_like(
            dt,
            spline_coef,
        )

        # ========================================================
        # 5. PHYSICAL TIME
        # ========================================================

        physical_time = (
            t_start
            +
            delta_t
        )

        # ========================================================
        # BRANCH A
        #
        # ORIGINAL CFO FLOW-MATCHING LOSS
        # ========================================================

        # --------------------------------------------------------
        # CFO stochastic conditional state
        #
        # I(t)
        # =
        # s(t)
        # +
        # gamma(t) epsilon
        # --------------------------------------------------------

        noisy_state = self.sample_conditional_path(
            tau_reshaped,
            spline_coef,
            eps,
            dt_reshaped,
        )

        # --------------------------------------------------------
        # Neural operator predicts CFO velocity
        #
        # N_theta(t, I(t))
        # --------------------------------------------------------

        cfo_prediction = self._model_apply(
            params,
            noisy_state,
            physical_time,
            condition,
        )

        # --------------------------------------------------------
        # Exact CFO velocity target
        #
        # dI/dt
        # --------------------------------------------------------

        cfo_target = self.compute_targets(
            spline_coef,
            tau_reshaped,
            dt_reshaped,
            eps,
        )

        # --------------------------------------------------------
        # CFO flow-matching loss
        # --------------------------------------------------------

        cfo_loss = jnp.mean(
            (
                cfo_prediction
                -
                cfo_target
            ) ** 2
        )

        # ========================================================
        # BRANCH B
        #
        # SHALLOW-WATER PHYSICS LOSS
        # ========================================================

        # --------------------------------------------------------
        # Clean physical spline state
        #
        # q = s(t)
        #
        # IMPORTANT:
        # no CFO noise is present here.
        # --------------------------------------------------------

        clean_state = self.spline_mean(
            tau_reshaped,
            spline_coef,
        )

        # --------------------------------------------------------
        # Ask the SAME neural operator for q_t
        #
        # q_t^NN = N_theta(t, q)
        # --------------------------------------------------------

        physics_q_t = self._model_apply(
            params,
            clean_state,
            physical_time,
            condition,
        )

        # --------------------------------------------------------
        # Explicit 2D SWE residual
        #
        # R
        # =
        # q_t
        # +
        # div(F(q))
        #
        # Current Phase:
        #
        # flat bathymetry b = 0
        # --------------------------------------------------------

        pde_loss = swe_physics_loss(
            q=clean_state,
            q_t=physics_q_t,
            dx=self.dx,
            dy=self.dy,
            g=self.gravity,
        )

        # ========================================================
        # TOTAL PI-CFO LOSS
        # ========================================================

        total_loss = (
            cfo_loss
            +
            self.lambda_pde
            *
            pde_loss
        )

        return (
            total_loss,
            cfo_loss,
            pde_loss,
        )

    # ============================================================
    # TRAINING LOSS EXPECTED BY train.py
    # ============================================================

    def loss_fn(
        self,
        params,
        batch,
    ):
        """Return scalar PI-CFO loss for JAX optimization."""

        (
            total_loss,
            _,
            _,
        ) = self.loss_components(
            params,
            batch,
        )

        return total_loss