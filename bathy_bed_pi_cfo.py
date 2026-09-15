"""
Bathymetry PI-CFO with an explicit bed-response loss.

Objective
---------
L_total
=
L_CFO
+
lambda_PDE * L_PDE
+
lambda_bed * L_bed

The bed-response loss isolates the effect of changing bathymetry
while keeping the physical state q and time t fixed:

    delta_qt_NN
    =
    N_theta(t, q, b)
    -
    N_theta(t, q, 0)

For the variable-bottom SWE, changing only b contributes directly
to momentum through the bed-slope source:

    delta(hu)_t = -S_x
    delta(hv)_t = -S_y

where S_x and S_y are calculated using the same well-balanced
discretization used by the existing SWE residual.

Only the momentum channels are included in L_bed because the
continuity equation has no direct bathymetry source term.
"""

from __future__ import annotations

import jax.numpy as jnp

from cfo import ContinuousFlowOperator

from utils.physics_swe_bathy import (
    swe_bathy_physics_loss,
    well_balanced_bed_source,
)


class BathymetryBedRegularizedPICFO(
    ContinuousFlowOperator
):
    """
    Bathymetry-conditioned PI-CFO with explicit bed-response matching.

    The model learns

        N_theta(t, q, b)
        =
        [h_t, (hu)_t, (hv)_t]

    using three objectives:

        1. CFO flow matching
        2. full variable-bottom SWE residual
        3. isolated bathymetry-response loss
    """

    def __init__(
        self,
        *,
        model,
        input_shape,
        condition_shape,
        gamma: float = 1e-5,
        spline_type: str = "quintic",
        lambda_pde: float = 0.03,
        lambda_bed: float = 0.1,
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

        self.lambda_bed = float(
            lambda_bed
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
    # BED-RESPONSE LOSS
    # ============================================================

    def bed_response_loss(
        self,
        params,
        q,
        physical_time,
        bathymetry,
        *,
        q_t_bathy=None,
    ):
        """
        Compare the network's response to changing bathymetry with
        the bed-slope contribution predicted by the SWE.

        q and physical_time are kept exactly fixed.

        Network response:

            delta q_t
            =
            N(t,q,b)
            -
            N(t,q,0)

        SWE expectation:

            delta(hu)_t = -S_x
            delta(hv)_t = -S_y

        Only interior momentum cells are compared.
        """

        # --------------------------------------------------------
        # Network derivative with the real bathymetry.
        #
        # Reuse q_t_bathy when already calculated for L_PDE.
        # --------------------------------------------------------

        if q_t_bathy is None:

            q_t_bathy = self._model_apply(
                params,
                q,
                physical_time,
                bathymetry,
            )

        # --------------------------------------------------------
        # Counterfactual flat-bed condition.
        #
        # IMPORTANT:
        # q and t are unchanged.
        # Only b changes.
        # --------------------------------------------------------

        zero_bathymetry = jnp.zeros_like(
            bathymetry
        )

        q_t_flat = self._model_apply(
            params,
            q,
            physical_time,
            zero_bathymetry,
        )

        # --------------------------------------------------------
        # What did the network say changed because of b?
        # --------------------------------------------------------

        delta_q_t = (
            q_t_bathy
            -
            q_t_flat
        )

        # --------------------------------------------------------
        # SWE well-balanced bed source.
        #
        # well_balanced_bed_source returns the LHS contributions:
        #
        #     +S_x = +g h b_x
        #     +S_y = +g h b_y
        #
        # Since:
        #
        #     q_t + flux divergence + source = 0
        #
        # the direct derivative contribution is:
        #
        #     delta(hu)_t = -S_x
        #     delta(hv)_t = -S_y
        # --------------------------------------------------------

        (
            source_x,
            source_y,
        ) = well_balanced_bed_source(
            q,
            bathymetry,
            dx=self.dx,
            dy=self.dy,
            g=self.gravity,
        )

        expected_momentum_change = jnp.stack(
            [
                -source_x,
                -source_y,
            ],
            axis=-1,
        )

        # --------------------------------------------------------
        # Network momentum response on the same interior cells.
        #
        # Channel:
        #
        # 0 = h_t
        # 1 = (hu)_t
        # 2 = (hv)_t
        # --------------------------------------------------------

        predicted_momentum_change = (
            delta_q_t[
                :,
                1:-1,
                1:-1,
                1:3,
            ]
        )

        # --------------------------------------------------------
        # Explicit magnitude + spatial-pattern error.
        # --------------------------------------------------------

        bed_loss = jnp.mean(
            (
                predicted_momentum_change
                -
                expected_momentum_change
            )
            ** 2
        )

        return bed_loss

    # ============================================================
    # ALL LOSS COMPONENTS
    # ============================================================

    def loss_components(
        self,
        params,
        batch,
    ):
        """
        Returns

            total_loss
            cfo_loss
            pde_loss
            bed_loss
        """

        (
            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps,
        ) = batch

        # ========================================================
        # SPLINE TIME
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

        tau_reshaped = self._reshape_time_like(
            tau,
            spline_coef,
        )

        dt_reshaped = self._reshape_time_like(
            dt,
            spline_coef,
        )

        physical_time = (
            t_start
            +
            delta_t
        )

        # ========================================================
        # BRANCH 1 — CFO FLOW MATCHING
        # ========================================================

        noisy_state = self.sample_conditional_path(
            tau_reshaped,
            spline_coef,
            eps,
            dt_reshaped,
        )

        cfo_prediction = self._model_apply(
            params,
            noisy_state,
            physical_time,
            bathymetry,
        )

        cfo_target = self.compute_targets(
            spline_coef,
            tau_reshaped,
            dt_reshaped,
            eps,
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
        # CLEAN PHYSICAL STATE
        # ========================================================

        clean_state = self.spline_mean(
            tau_reshaped,
            spline_coef,
        )

        # ========================================================
        # NETWORK VECTOR FIELD ON CLEAN STATE
        # ========================================================

        physics_q_t = self._model_apply(
            params,
            clean_state,
            physical_time,
            bathymetry,
        )

        # ========================================================
        # BRANCH 2 — FULL VARIABLE-BOTTOM SWE LOSS
        # ========================================================

        pde_loss = swe_bathy_physics_loss(
            q=clean_state,
            q_t=physics_q_t,
            bathymetry=bathymetry,
            dx=self.dx,
            dy=self.dy,
            g=self.gravity,
        )

        # ========================================================
        # BRANCH 3 — EXPLICIT BED-RESPONSE LOSS
        # ========================================================

        bed_loss = self.bed_response_loss(
            params,
            clean_state,
            physical_time,
            bathymetry,
            q_t_bathy=physics_q_t,
        )

        # ========================================================
        # TOTAL
        # ========================================================

        total_loss = (
            cfo_loss
            +
            self.lambda_pde
            *
            pde_loss
            +
            self.lambda_bed
            *
            bed_loss
        )

        return (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
        )

    # ============================================================
    # SCALAR LOSS
    # ============================================================

    def loss_fn(
        self,
        params,
        batch,
    ):

        (
            total_loss,
            _,
            _,
            _,
        ) = self.loss_components(
            params,
            batch,
        )

        return total_loss


__all__ = [
    "BathymetryBedRegularizedPICFO",
]