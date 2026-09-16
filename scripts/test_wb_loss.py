"""
Smoke test for the explicit well-balanced loss.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from models.fno import FNO2d

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)


def gaussian_bathymetry(
    batch=2,
    n=32,
):

    x = jnp.linspace(
        -2.5,
        2.5,
        n,
    )

    y = jnp.linspace(
        -2.5,
        2.5,
        n,
    )

    X, Y = jnp.meshgrid(
        x,
        y,
        indexing="ij",
    )

    b = (
        0.2
        *
        jnp.exp(
            -(
                X**2
                +
                Y**2
            )
            /
            (
                2.0
                *
                0.5**2
            )
        )
    )

    b = b[
        None,
        ...,
        None
    ]

    return jnp.repeat(
        b,
        batch,
        axis=0,
    )


def main():

    model = FNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        use_condition=True,
        use_time=True,
    )

    method = (
        WellBalancedBathymetryBedPICFO(
            model=model,
            input_shape=(
                32,
                32,
                3,
            ),
            condition_shape=(
                32,
                32,
                1,
            ),
            gamma=1e-5,
            spline_type="quintic",
            lambda_pde=0.03,
            lambda_bed=0.70,
            lambda_wb=0.03,
            wb_eta0=1.5,
            dx=0.15625,
            dy=0.15625,
            gravity=1.0,
        )
    )

    state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
    )

    b = gaussian_bathymetry()

    t = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    loss = method.well_balanced_loss(
        state.params,
        b,
        t,
    )

    print()
    print("============================")
    print("WB LOSS SMOKE TEST")
    print("============================")
    print("WB loss:", float(loss))
    print(
        "finite:",
        bool(
            jnp.isfinite(
                loss
            )
        ),
    )

    grads = jax.grad(
        lambda p:
        method.well_balanced_loss(
            p,
            b,
            t,
        )
    )(
        state.params
    )

    finite_gradients = all(
        bool(
            jnp.all(
                jnp.isfinite(
                    leaf
                )
            )
        )
        for leaf
        in jax.tree_util.tree_leaves(
            grads
        )
    )

    print(
        "gradients finite:",
        finite_gradients,
    )

    if not bool(
        jnp.isfinite(
            loss
        )
    ):
        raise RuntimeError(
            "WB loss is not finite."
        )

    if not finite_gradients:
        raise RuntimeError(
            "WB gradients are not finite."
        )

    print("============================")
    print("WB LOSS TEST PASSED")
    print("============================")


if __name__ == "__main__":
    main()