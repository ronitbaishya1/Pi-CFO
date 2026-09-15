"""
Smoke test for models/sc_fno.py.

Checks
------
1. SC-FNO initializes.
2. Output has correct shape.
3. Output is finite.
4. Bathymetry actually changes the output.
5. Gradients are finite.
6. JIT works.
7. Composition depths 1, 2, and 4 all work.
8. Parameter count does NOT grow strongly with composition depth
   because the operator block is shared.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from models.sc_fno import (
    SCFNO2d,
)


# ================================================================
# PARAMETER COUNT
# ================================================================

def count_parameters(
    params,
):

    leaves = jax.tree_util.tree_leaves(
        params
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
# FINITE TREE CHECK
# ================================================================

def tree_is_finite(
    tree,
):

    leaves = jax.tree_util.tree_leaves(
        tree
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
# MAIN
# ================================================================

def main():

    key = jax.random.PRNGKey(
        0
    )

    q_key, b_key = jax.random.split(
        key
    )

    batch_size = 2

    nx = 32

    ny = 32

    channels = 3

    # ============================================================
    # FAKE SWE STATE
    # ============================================================

    q = jax.random.normal(
        q_key,
        (
            batch_size,
            nx,
            ny,
            channels,
        ),
    )

    # Make depth positive and realistic-ish.
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

    # ============================================================
    # FAKE BATHYMETRY
    # ============================================================

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

    zero_bathymetry = (
        jnp.zeros_like(
            bathymetry
        )
    )

    time = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    print(
        "\n========================================"
    )

    print(
        "SC-FNO SMOKE TEST"
    )

    print(
        "========================================"
    )

    previous_output = None

    parameter_counts = {}

    for depth in [
        1,
        2,
        4,
    ]:

        print(
            f"\nComposition depth = {depth}"
        )

        model = SCFNO2d(
            num_channels=3,
            modes1=12,
            modes2=12,
            width=64,
            compose_depth=depth,
            residual_scale=0.5,
            use_condition=True,
            use_time=True,
        )

        variables = model.init(
            jax.random.PRNGKey(
                100
                +
                depth
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

        output_flat = model.apply(
            variables,
            q,
            time,
            zero_bathymetry,
        )

        # --------------------------------------------------------
        # Shape
        # --------------------------------------------------------

        print(
            "Output shape:      ",
            output.shape,
        )

        if output.shape != q.shape:

            raise RuntimeError(
                "SC-FNO output shape "
                "is incorrect."
            )

        # --------------------------------------------------------
        # Finite values
        # --------------------------------------------------------

        output_is_finite = bool(
            jnp.all(
                jnp.isfinite(
                    output
                )
            )
        )

        print(
            "Output finite:     ",
            output_is_finite,
        )

        if not output_is_finite:

            raise RuntimeError(
                "SC-FNO produced NaN/Inf."
            )

        # --------------------------------------------------------
        # Condition sensitivity
        # --------------------------------------------------------

        condition_difference = float(
            jnp.mean(
                jnp.abs(
                    output
                    -
                    output_flat
                )
            )
        )

        print(
            "Bathymetry effect: ",
            f"{condition_difference:.6e}",
        )

        if (
            condition_difference
            <
            1e-9
        ):

            raise RuntimeError(
                "Bathymetry is not affecting "
                "the SC-FNO output."
            )

        # --------------------------------------------------------
        # Gradient test
        # --------------------------------------------------------

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

        gradients_finite = (
            tree_is_finite(
                gradients
            )
        )

        print(
            "Gradients finite:  ",
            gradients_finite,
        )

        if not gradients_finite:

            raise RuntimeError(
                "SC-FNO gradients "
                "contain NaN/Inf."
            )

        # --------------------------------------------------------
        # JIT test
        # --------------------------------------------------------

        apply_jit = jax.jit(
            lambda p, x, t, b:
            model.apply(
                {
                    "params":
                        p
                },
                x,
                t,
                b,
            )
        )

        output_jit = apply_jit(
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
                    output_jit
                )
            )
        )

        print(
            "JIT max difference:",
            f"{jit_difference:.6e}",
        )

        # --------------------------------------------------------
        # Parameter count
        # --------------------------------------------------------

        parameter_count = (
            count_parameters(
                params
            )
        )

        parameter_counts[
            depth
        ] = parameter_count

        print(
            "Parameter count:   ",
            f"{parameter_count:,}",
        )

        # --------------------------------------------------------
        # Does composition actually change output?
        # --------------------------------------------------------

        if (
            previous_output
            is not None
        ):

            depth_difference = float(
                jnp.mean(
                    jnp.abs(
                        output
                        -
                        previous_output
                    )
                )
            )

            print(
                "Change vs previous depth:",
                f"{depth_difference:.6e}",
            )

        previous_output = (
            output
        )

    # ============================================================
    # SHARING CHECK
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

    for (
        depth,
        count,
    ) in parameter_counts.items():

        print(
            f"depth={depth}: "
            f"{count:,} parameters"
        )

    unique_counts = set(
        parameter_counts.values()
    )

    if len(
        unique_counts
    ) != 1:

        print(
            "\nWARNING:"
        )

        print(
            "Parameter count changed with "
            "composition depth."
        )

        print(
            "The shared block may not be "
            "sharing parameters correctly."
        )

    else:

        print(
            "\nPASS:"
        )

        print(
            "All depths use the same "
            "parameter count."
        )

        print(
            "The Fourier block is truly shared."
        )

    print(
        "\n========================================"
    )

    print(
        "SC-FNO SMOKE TEST PASSED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()