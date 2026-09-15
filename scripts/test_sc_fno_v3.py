"""
Smoke test for SCFNO2dV3.

Checks
------
1. K=1,2,3,4 initialize.
2. Correct output shape.
3. Finite output.
4. Finite gradients.
5. Bathymetry affects prediction.
6. JIT works.
7. Parameter count is identical for K=1,2,3,4.
8. Checkpoint parameter trees are compatible across depths.

The last point is essential for Train-and-Unroll.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


# ================================================================
# ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ================================================================
# IMPORT
# ================================================================

from models.sc_fno_v3 import (
    SCFNO2dV3,
)


# ================================================================
# COUNT PARAMETERS
# ================================================================

def count_parameters(
    params,
):

    leaves = (
        jax.tree_util.tree_leaves(
            params
        )
    )

    return int(
        sum(
            np.prod(
                leaf.shape
            )
            for leaf
            in leaves
        )
    )


# ================================================================
# FINITE TREE
# ================================================================

def tree_is_finite(
    tree,
):

    leaves = (
        jax.tree_util.tree_leaves(
            tree
        )
    )

    return all(
        bool(
            jnp.all(
                jnp.isfinite(
                    leaf
                )
            )
        )
        for leaf
        in leaves
    )


# ================================================================
# TREE SHAPES
# ================================================================

def tree_shapes(
    tree,
):

    return [
        tuple(
            leaf.shape
        )
        for leaf
        in jax.tree_util.tree_leaves(
            tree
        )
    ]


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "SC-FNO-v3 SMOKE TEST"
    )

    print(
        "========================================"
    )

    # ============================================================
    # INPUT
    # ============================================================

    batch_size = 2

    nx = 32

    ny = 32

    q_key = jax.random.PRNGKey(
        1
    )

    b_key = jax.random.PRNGKey(
        2
    )

    q = jax.random.normal(
        q_key,
        (
            batch_size,
            nx,
            ny,
            3,
        ),
    )

    # ------------------------------------------------------------
    # Make h positive
    # ------------------------------------------------------------

    q = q.at[
        ...,
        0
    ].set(
        1.2
        +
        0.05
        *
        q[
            ...,
            0
        ]
    )

    bathymetry = (
        0.15
        *
        jax.nn.sigmoid(
            jax.random.normal(
                b_key,
                (
                    batch_size,
                    nx,
                    ny,
                    1,
                ),
            )
        )
    )

    zero_bathymetry = jnp.zeros_like(
        bathymetry
    )

    time = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # DEPTH TEST
    # ============================================================

    parameter_counts = {}

    parameter_shapes = {}

    for depth in [
        1,
        2,
        3,
        4,
    ]:

        print(
            "\n----------------------------------------"
        )

        print(
            f"Composition depth K={depth}"
        )

        print(
            "----------------------------------------"
        )

        model = SCFNO2dV3(
            num_channels=3,
            modes1=12,
            modes2=12,
            width=64,
            compose_depth=depth,
            alpha=0.25,
            use_condition=True,
            use_time=True,
        )

        variables = model.init(
            jax.random.PRNGKey(
                100
            ),
            q,
            time,
            bathymetry,
        )

        params = variables[
            "params"
        ]

        output = model.apply(
            variables,
            q,
            time,
            bathymetry,
        )

        flat_output = model.apply(
            variables,
            q,
            time,
            zero_bathymetry,
        )

        # ========================================================
        # SHAPE
        # ========================================================

        print(
            "Output shape:      ",
            output.shape,
        )

        if output.shape != q.shape:

            raise RuntimeError(
                f"K={depth}: wrong output shape."
            )

        # ========================================================
        # FINITE OUTPUT
        # ========================================================

        finite_output = bool(
            jnp.all(
                jnp.isfinite(
                    output
                )
            )
        )

        print(
            "Output finite:      ",
            finite_output,
        )

        if not finite_output:

            raise RuntimeError(
                f"K={depth}: NaN/Inf output."
            )

        # ========================================================
        # BATHYMETRY EFFECT
        # ========================================================

        bathymetry_effect = float(
            jnp.mean(
                jnp.abs(
                    output
                    -
                    flat_output
                )
            )
        )

        print(
            "Bathymetry effect:  ",
            f"{bathymetry_effect:.6e}",
        )

        if bathymetry_effect < 1e-10:

            raise RuntimeError(
                f"K={depth}: bathymetry "
                f"does not affect prediction."
            )

        # ========================================================
        # GRADIENT
        # ========================================================

        def loss_fn(
            test_params,
        ):

            prediction = model.apply(
                {
                    "params":
                        test_params
                },
                q,
                time,
                bathymetry,
            )

            return jnp.mean(
                prediction**2
            )

        gradients = jax.grad(
            loss_fn
        )(
            params
        )

        gradients_finite = tree_is_finite(
            gradients
        )

        print(
            "Gradients finite:   ",
            gradients_finite,
        )

        if not gradients_finite:

            raise RuntimeError(
                f"K={depth}: gradients "
                f"contain NaN/Inf."
            )

        # ========================================================
        # JIT
        # ========================================================

        @jax.jit
        def jit_apply(
            test_params,
            q_input,
            t_input,
            b_input,
        ):

            return model.apply(
                {
                    "params":
                        test_params
                },
                q_input,
                t_input,
                b_input,
            )

        jit_output = jit_apply(
            params,
            q,
            time,
            bathymetry,
        )

        jit_difference = float(
            jnp.max(
                jnp.abs(
                    output
                    -
                    jit_output
                )
            )
        )

        print(
            "JIT difference:     ",
            f"{jit_difference:.6e}",
        )

        # ========================================================
        # PARAMETER COUNT
        # ========================================================

        parameter_count = count_parameters(
            params
        )

        parameter_counts[
            depth
        ] = parameter_count

        parameter_shapes[
            depth
        ] = tree_shapes(
            params
        )

        print(
            "Parameter count:    ",
            f"{parameter_count:,}",
        )

    # ============================================================
    # PARAMETER SHARING CHECK
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "PARAMETER SHARING CHECK"
    )

    print(
        "========================================"
    )

    for depth in [
        1,
        2,
        3,
        4,
    ]:

        print(
            f"K={depth}: "
            f"{parameter_counts[depth]:,}"
        )

    unique_counts = set(
        parameter_counts.values()
    )

    if len(
        unique_counts
    ) != 1:

        raise RuntimeError(
            "Parameter count changes "
            "with composition depth."
        )

    print(
        "\nPASS: parameter count is "
        "identical for K=1,2,3,4."
    )

    # ============================================================
    # PARAMETER TREE COMPATIBILITY
    # ============================================================

    reference_shapes = parameter_shapes[
        1
    ]

    for depth in [
        2,
        3,
        4,
    ]:

        if (
            parameter_shapes[
                depth
            ]
            !=
            reference_shapes
        ):

            raise RuntimeError(
                f"K={depth} parameter tree "
                f"is incompatible with K=1."
            )

    print(
        "PASS: parameter trees are "
        "compatible across all depths."
    )

    print(
        "\nThis means Train-and-Unroll "
        "checkpoint transfer is possible."
    )

    print(
        "\n========================================"
    )

    print(
        "SC-FNO-v3 SMOKE TEST PASSED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()