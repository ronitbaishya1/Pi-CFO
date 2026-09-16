"""
Well-balanced Bed-PI-CFO for variable-bottom shallow-water dynamics.

Objective
---------
L_total
=
L_CFO
+
lambda_PDE * L_PDE
+
lambda_bed * L_bed
+
lambda_WB * L_WB

where

L_WB = mean( |N_theta(t, q_eq, b)|^2 )

and

q_eq = [eta0 - b, 0, 0]

is a lake-at-rest state.

The class deliberately subclasses the existing
BathymetryBedRegularizedPICFO so that CFO, PDE loss and bed-response
loss remain exactly the same as the established baseline.
"""

from __future__ import annotations

import jax.numpy as jnp

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from utils.physics_swe_bathy import (
    lake_at_rest_state,
)


class WellBalancedBathymetryBedPICFO(
    BathymetryBedRegularizedPICFO
):
    """
    Bed-PI-CFO + explicit lake-at-rest loss.
    """

    def __init__(
        self,
        *args,
        lambda_wb: float = 0.03,
        wb_eta0: float = 1.5,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.lambda_wb = float(
            lambda_wb
        )

        self.wb_eta0 = float(
            wb_eta0
        )

        if self.lambda_wb < 0.0:

            raise ValueError(
                "lambda_wb must be >= 0."
            )

    # ============================================================
    # WELL-BALANCED LOSS
    # ============================================================

    def well_balanced_loss(
        self,
        params,
        bathymetry,
        time,
    ):
        """
        Construct a lake-at-rest state over the supplied bathymetry
        and require the learned vector field to be zero.

        Parameters
        ----------
        bathymetry:
            (B,H,W,1)

        time:
            (B,)

        Returns
        -------
        scalar JAX loss
        """

        q_equilibrium = (
            lake_at_rest_state(
                bathymetry,
                eta0=(
                    self.wb_eta0
                ),
            )
        )

        q_t_equilibrium = (
            self._model_apply(
                params,
                q_equilibrium,
                time,
                bathymetry,
            )
        )

        return jnp.mean(
            q_t_equilibrium**2
        )

    # ============================================================
    # COMPLETE LOSS
    # ============================================================

    def loss_components(
        self,
        params,
        batch,
    ):
        """
        Expected conditioned CFO batch:

            spline_coef,
            bathymetry,
            t_start,
            t_end,
            delta_t,
            eps

        The parent returns at least:

            total,
            CFO,
            PDE,
            bed

        Extra parent diagnostics, if present, are preserved.
        """

        parent_components = (
            super().loss_components(
                params,
                batch,
            )
        )

        if len(
            parent_components
        ) < 4:

            raise RuntimeError(
                "Expected parent Bed-PI-CFO "
                "loss_components() to return "
                "at least total, CFO, PDE and bed."
            )

        parent_total = (
            parent_components[
                0
            ]
        )

        bathymetry = batch[
            1
        ]

        t_start = jnp.reshape(
            batch[
                2
            ],
            (-1,),
        )

        t_end = jnp.reshape(
            batch[
                3
            ],
            (-1,),
        )

        # Arbitrary physical time inside the same interval.
        # A true steady state must have q_t=0 at every time.
        wb_time = (
            0.5
            *
            (
                t_start
                +
                t_end
            )
        )

        wb_loss = (
            self.well_balanced_loss(
                params,
                bathymetry,
                wb_time,
            )
        )

        total_loss = (
            parent_total
            +
            self.lambda_wb
            *
            wb_loss
        )

        # Keep parent's diagnostics and append WB.
        return (
            total_loss,
            *parent_components[
                1:
            ],
            wb_loss,
        )


__all__ = [
    "WellBalancedBathymetryBedPICFO",
]