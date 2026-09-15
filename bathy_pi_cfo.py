"""Physics-Informed CFO for 2D shallow water over variable bathymetry."""

from __future__ import annotations

import jax.numpy as jnp

from cfo import ContinuousFlowOperator

from utils.physics_swe_bathy import (
    swe_bathy_physics_loss,
)


class BathymetryPhysicsInformedCFO(
    ContinuousFlowOperator
):
    """
    Bathymetry-conditioned Physics-Informed CFO.

    The neural operator is

        N_theta(t, q, b)

    where

        q = [h, hu, hv]

    and

        b = bathymetry.

    CFO branch:

        L_CFO
        =
        ||N_theta(t, I(t), b) - dI/dt||^2

    Physics branch:

        L_PDE
        =
        ||R_SWE(
            s(t),
            N_theta(t, s(t), b),
            b
        )||^2

    Total:

        L_total
        =
        L_CFO
        +
        lambda_pde * L_PDE

    Physics is applied to the clean spline state s(t),
    not to the stochastic CFO state I(t).
    """

    def __init__(
        self,
        *,
        model,
        input_shape,
        condition_shape,
        gamma: float = 1e-5,
        spline_type: str = "quintic",
        lambda_pde: float = 0.1,
        dx: float = 0.15625,
        dy: float = 0.15625,
        gravity: float = 1.0,
    ):

        super().__init__(
            model=model,
            input_shape=input_shape,
            gamma=gamma,
            spline_type=spline_type,
            use_condition=True,
            condition_shape=condition_shape,
        )

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

    def loss_components(
        self,
        params,
        batch,
    ):

        (
            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps,
        ) = batch

        # ========================================================
        # 1. SPLINE INTERVAL
        # ========================================================

        dt = (
            t_end
            -
            t_start
        )

        tau = (
            delta_t
            /
            dt
        )

        tau_reshaped = (
            self._reshape_time_like(
                tau,
                spline_coef,
            )
        )

        dt_reshaped = (
            self._reshape_time_like(
                dt,
                spline_coef,
            )
        )

        physical_time = (
            t_start
            +
            delta_t
        )

        # ========================================================
        # 2. CFO BRANCH
        #
        # N_theta(t, I(t), b)
        # ========================================================

        noisy_state = (
            self.sample_conditional_path(
                tau_reshaped,
                spline_coef,
                eps,
                dt_reshaped,
            )
        )

        cfo_prediction = (
            self._model_apply(
                params,
                noisy_state,
                physical_time,
                bathymetry,
            )
        )

        cfo_target = (
            self.compute_targets(
                spline_coef,
                tau_reshaped,
                dt_reshaped,
                eps,
            )
        )

        cfo_loss = jnp.mean(
            (
                cfo_prediction
                -
                cfo_target
            )
            ** 2
        )

        # ========================================================
        # 3. CLEAN PHYSICAL STATE
        #
        # q = s(t)
        # ========================================================

        clean_state = (
            self.spline_mean(
                tau_reshaped,
                spline_coef,
            )
        )

        # ========================================================
        # 4. PHYSICS VECTOR FIELD
        #
        # q_t = N_theta(t, q, b)
        # ========================================================

        physics_q_t = (
            self._model_apply(
                params,
                clean_state,
                physical_time,
                bathymetry,
            )
        )

        # ========================================================
        # 5. VARIABLE-BOTTOM SWE RESIDUAL
        # ========================================================

        pde_loss = (
            swe_bathy_physics_loss(
                q=clean_state,
                q_t=physics_q_t,
                bathymetry=bathymetry,
                dx=self.dx,
                dy=self.dy,
                g=self.gravity,
            )
        )

        # ========================================================
        # 6. TOTAL LOSS
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

    def loss_fn(
        self,
        params,
        batch,
    ):

        (
            total_loss,
            _,
            _,
        ) = self.loss_components(
            params,
            batch,
        )

        return total_loss


__all__ = [
    "BathymetryPhysicsInformedCFO",
]