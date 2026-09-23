"""
Tangent-consistent Well-Balanced Bed-PI-CFO.

Base objective
--------------

The established model already uses

    L_base
    =
    L_CFO
    +
    lambda_PDE * L_PDE
    +
    lambda_bed * L_bed
    +
    lambda_WB * L_WB.

This class preserves all four terms exactly and adds

    lambda_tangent * L_tangent.

Tangent consistency
-------------------

For a physical state q, bathymetry b, time t and perturbation v,

    J_NN(q,b,t) v

is compared against

    J_SWE(q,b) v.

The SWE tangent comes from the SAME discrete well-balanced SWE
operator already used by the PDE loss.

We do NOT force the dynamics to be dissipative.

Instead, we require the neural vector field to respond to local
state perturbations similarly to the physical SWE vector field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from utils.tangent_swe_bathy import (
    swe_bathy_rhs_interior,
    prepare_tangent_direction,
)


class TangentConsistentWellBalancedBathymetryBedPICFO(
    WellBalancedBathymetryBedPICFO
):
    """
    Existing WB-Bed-PI-CFO plus directional SWE tangent consistency.
    """

    def __init__(
        self,
        *args,
        lambda_tangent: float = 0.001,
        tangent_eps: float = 1.0e-12,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.lambda_tangent = float(
            lambda_tangent
        )

        self.tangent_eps = float(
            tangent_eps
        )

        if self.lambda_tangent < 0.0:

            raise ValueError(
                "lambda_tangent must be >= 0."
            )

        if self.tangent_eps <= 0.0:

            raise ValueError(
                "tangent_eps must be > 0."
            )

    # =================================================================
    # TANGENT RESPONSE
    # =================================================================

    def tangent_consistency_components(
        self,
        params,
        q,
        physical_time,
        bathymetry,
        tangent_direction,
    ):
        """
        Calculate directional tangent consistency.

        Returns
        -------
        tangent_loss:
            symmetric normalized discrepancy

        tangent_cosine:
            directional alignment between J_NN v and J_SWE v

        tangent_gain_ratio:
            ||J_NN v|| / ||J_SWE v||

        The first quantity enters the training objective.

        Cosine and gain ratio are diagnostics only.
        """

        # -------------------------------------------------------------
        # PREPARE DIRECTION
        #
        # The training script already creates a smooth normalized
        # perturbation, but preparing again here guarantees consistent
        # boundary treatment and normalization.
        # -------------------------------------------------------------

        direction = prepare_tangent_direction(
            tangent_direction,
            eps=self.tangent_eps,
        )

        # =============================================================
        # NEURAL TANGENT RESPONSE
        #
        # q is the differentiated variable.
        #
        # t and b remain fixed.
        # =============================================================

        def neural_vector_field(
            q_input,
        ):

            return self._model_apply(
                params,
                q_input,
                physical_time,
                bathymetry,
            )

        (
            _,
            neural_jvp_full,
        ) = jax.jvp(
            neural_vector_field,
            (
                q,
            ),
            (
                direction,
            ),
        )

        # -------------------------------------------------------------
        # The physical SWE RHS is defined on the common interior grid.
        # Compare the neural response on exactly the same cells.
        # -------------------------------------------------------------

        neural_jvp = neural_jvp_full[
            :,
            1:-1,
            1:-1,
            :,
        ]

        # =============================================================
        # PHYSICAL SWE TANGENT RESPONSE
        # =============================================================

        def physical_vector_field(
            q_input,
        ):

            return swe_bathy_rhs_interior(
                q_input,
                bathymetry,
                dx=self.dx,
                dy=self.dy,
                g=self.gravity,
            )

        (
            _,
            physical_jvp,
        ) = jax.jvp(
            physical_vector_field,
            (
                q,
            ),
            (
                direction,
            ),
        )

        # -------------------------------------------------------------
        # No trainable parameters exist in the physical branch.
        # Stop-gradient makes that explicit and reduces unnecessary
        # differentiation bookkeeping.
        # -------------------------------------------------------------

        physical_jvp = jax.lax.stop_gradient(
            physical_jvp
        )

        # =============================================================
        # PER-SAMPLE MAGNITUDES
        # =============================================================

        reduction_axes = (
            1,
            2,
            3,
        )

        neural_squared = jnp.sum(
            neural_jvp**2,
            axis=reduction_axes,
        )

        physical_squared = jnp.sum(
            physical_jvp**2,
            axis=reduction_axes,
        )

        difference_squared = jnp.sum(
            (
                neural_jvp
                -
                physical_jvp
            )
            ** 2,
            axis=reduction_axes,
        )

        # =============================================================
        # TANGENT LOSS
        #
        #                  ||J_NN v - J_SWE v||^2
        #
        # L_tan = -----------------------------------------------
        #         ||J_NN v||^2 + ||J_SWE v||^2 + eps
        #
        #
        # 0 = perfect tangent agreement.
        #
        # This is exactly the symmetric normalized metric we used
        # during Stage A.
        # =============================================================

        tangent_error_per_sample = (
            difference_squared
            /
            (
                neural_squared
                +
                physical_squared
                +
                self.tangent_eps
            )
        )

        tangent_loss = jnp.mean(
            tangent_error_per_sample
        )

        # =============================================================
        # COSINE SIMILARITY
        # =============================================================

        dot_product = jnp.sum(
            neural_jvp
            *
            physical_jvp,
            axis=reduction_axes,
        )

        neural_norm = jnp.sqrt(
            neural_squared
            +
            self.tangent_eps
        )

        physical_norm = jnp.sqrt(
            physical_squared
            +
            self.tangent_eps
        )

        tangent_cosine_per_sample = (
            dot_product
            /
            (
                neural_norm
                *
                physical_norm
                +
                self.tangent_eps
            )
        )

        tangent_cosine = jnp.mean(
            tangent_cosine_per_sample
        )

        # =============================================================
        # GAIN RATIO
        #
        # Desired value:
        #
        #     approximately 1.
        # =============================================================

        tangent_gain_ratio_per_sample = (
            neural_norm
            /
            (
                physical_norm
                +
                self.tangent_eps
            )
        )

        tangent_gain_ratio = jnp.mean(
            tangent_gain_ratio_per_sample
        )

        return (
            tangent_loss,
            tangent_cosine,
            tangent_gain_ratio,
        )

    # =================================================================
    # COMPLETE LOSS
    # =================================================================

    def loss_components(
        self,
        params,
        batch,
    ):
        """
        Expected batch
        --------------

        (
            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps,
            tangent_direction,
        )

        Returns
        -------

        total
        CFO
        PDE
        bed
        WB
        tangent
        tangent_cosine
        tangent_gain_ratio
        """

        if len(
            batch
        ) != 7:

            raise ValueError(
                "Tangent-consistent WB-Bed-PI-CFO "
                "expects a 7-element training batch."
            )

        (
            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps,
            tangent_direction,
        ) = batch

        # =============================================================
        # EXISTING LOSS — UNCHANGED
        # =============================================================

        base_batch = (
            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps,
        )

        parent_components = (
            super().loss_components(
                params,
                base_batch,
            )
        )

        if len(
            parent_components
        ) != 5:

            raise RuntimeError(
                "Expected the existing "
                "WellBalancedBathymetryBedPICFO "
                "to return exactly:\n"
                "(total, CFO, PDE, bed, WB)."
            )

        (
            parent_total,
            cfo_loss,
            pde_loss,
            bed_loss,
            wb_loss,
        ) = parent_components

        # =============================================================
        # CLEAN SPLINE STATE
        #
        # This is the SAME physical state used by the current PDE and
        # bed losses.
        # =============================================================

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

        clean_state = self.spline_mean(
            tau_reshaped,
            spline_coef,
        )

        physical_time = (
            t_start
            +
            delta_t
        )

        # =============================================================
        # TANGENT CONSISTENCY
        # =============================================================

        (
            tangent_loss,
            tangent_cosine,
            tangent_gain_ratio,
        ) = self.tangent_consistency_components(
            params,
            clean_state,
            physical_time,
            bathymetry,
            tangent_direction,
        )

        # =============================================================
        # COMPLETE OBJECTIVE
        # =============================================================

        total_loss = (
            parent_total
            +
            self.lambda_tangent
            *
            tangent_loss
        )

        return (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            wb_loss,
            tangent_loss,
            tangent_cosine,
            tangent_gain_ratio,
        )

    # =================================================================
    # SCALAR LOSS
    # =================================================================

    def loss_fn(
        self,
        params,
        batch,
    ):

        components = self.loss_components(
            params,
            batch,
        )

        return components[
            0
        ]


__all__ = [
    "TangentConsistentWellBalancedBathymetryBedPICFO",
]