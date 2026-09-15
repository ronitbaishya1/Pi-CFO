"""
Smoke test for SC-FNO-v2.

Checks
------
1. Original FNO works.
2. SC-FNO-v2 K=2 works.
3. SC-FNO-v2 K=4 works.
4. Correct output shape.
5. Finite outputs.
6. Finite gradients.
7. Bathymetry changes output.
8. JIT works.
9. K=2 and K=4 have exactly the SAME number of parameters.
10. Compare SC-FNO-v2 parameter count with original FNO.
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
# IMPORTS
# ================================================================

from models.fno import (
    FNO2d,
)

from models.sc_fno_v2 import (
    SCFNO2dV2,
)


# ================================================================
# PARAMETER COUNT
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
# TEST ONE MODEL
# ================================================================

def test_model(
    *,
    model,
    name,
    q,
    t,
    bathymetry,
):

    zero_bathymetry = (
        jnp.zeros_like(
            bathymetry
        )
    )

    init_key = jax.random.PRNGKey(
        abs(
            hash(
                name
            )
        )
        %
        100000
    )

    # ============================================================
    # INITIALIZE
    # ============================================================

    variables = model.init(
        init_key,
        q,
        t,
        bathymetry,
    )

    params = variables[
        "params"
    ]

    # ============================================================
    # NORMAL OUTPUT
    # ============================================================

    output = model.apply(
        variables,
        q,
        t,
        bathymetry,
    )

    # ============================================================
    # FLAT-BED OUTPUT
    # ============================================================

    flat_output = model.apply(
        variables,
        q,
        t,
        zero_bathymetry,
    )

    # ============================================================
    # SHAPE
    # ============================================================

    expected_shape = (
        q.shape
    )

    if (
        output.shape
        !=
        expected_shape
    ):

        raise RuntimeError(
            f"{name}: wrong output shape. "
            f"Expected {expected_shape}, "
            f"got {output.shape}."
        )

    # ============================================================
    # FINITE OUTPUT
    # ============================================================

    if not bool(
        jnp.all(
            jnp.isfinite(
                output
            )
        )
    ):

        raise RuntimeError(
            f"{name}: output "
            f"contains NaN/Inf."
        )

    # ============================================================
    # CONDITION SENSITIVITY
    # ============================================================

    condition_effect = float(
        jnp.mean(
            jnp.abs(
                output
                -
                flat_output
            )
        )
    )

    if (
        condition_effect
        <
        1e-10
    ):

        raise RuntimeError(
            f"{name}: bathymetry "
            f"does not affect output."
        )

    # ============================================================
    # GRADIENT CHECK
    # ============================================================

    def loss_fn(
        test_params,
    ):

        prediction = model.apply(
            {
                "params":
                    test_params
            },
            q,
            t,
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

    if not tree_is_finite(
        gradients
    ):

        raise RuntimeError(
            f"{name}: gradients "
            f"contain NaN/Inf."
        )

    # ============================================================
    # JIT
    # ============================================================

    @jax.jit
    def apply_jit(
        test_params,
        q_input,
        t_input,
        bathymetry_input,
    ):

        return model.apply(
            {
                "params":
                    test_params
            },
            q_input,
            t_input,
            bathymetry_input,
        )

    output_jit = apply_jit(
        params,
        q,
        t,
        bathymetry,
    )

    jit_error = float(
        jnp.max(
            jnp.abs(
                output_jit
                -
                output
            )
        )
    )

    parameter_count = (
        count_parameters(
            params
        )
    )

    print(
        "\n----------------------------------------"
    )

    print(
        name
    )

    print(
        "----------------------------------------"
    )

    print(
        "Output shape:       ",
        output.shape,
    )

    print(
        "Output finite:       True"
    )

    print(
        "Gradients finite:    True"
    )

    print(
        "Bathymetry effect:  ",
        f"{condition_effect:.6e}",
    )

    print(
        "JIT max difference: ",
        f"{jit_error:.6e}",
    )

    print(
        "Parameter count:    ",
        f"{parameter_count:,}",
    )

    return {
        "parameter_count":
            parameter_count,

        "output":
            output,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "SC-FNO-v2 SMOKE TEST"
    )

    print(
        "========================================"
    )

    # ============================================================
    # INPUTS
    # ============================================================

    batch_size = 2

    nx = 32

    ny = 32

    channels = 3

    q_key = jax.random.PRNGKey(
        10
    )

    b_key = jax.random.PRNGKey(
        20
    )

    q = jax.random.normal(
        q_key,
        (
            batch_size,
            nx,
            ny,
            channels,
        ),
    )

    # Positive water depth
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

    # Small random positive bathymetry
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

    t = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    # ============================================================
    # ORIGINAL FNO
    # ============================================================

    original_fno = FNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        use_condition=True,
        use_time=True,
    )

    original_result = test_model(
        model=original_fno,
        name="Original FNO2d",
        q=q,
        t=t,
        bathymetry=bathymetry,
    )

    # ============================================================
    # SC K=2
    # ============================================================

    sc_k2 = SCFNO2dV2(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        compose_depth=2,
        use_condition=True,
        use_time=True,
    )

    k2_result = test_model(
        model=sc_k2,
        name="SC-FNO-v2 K=2",
        q=q,
        t=t,
        bathymetry=bathymetry,
    )

    # ============================================================
    # SC K=4
    # ============================================================

    sc_k4 = SCFNO2dV2(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        compose_depth=4,
        use_condition=True,
        use_time=True,
    )

    k4_result = test_model(
        model=sc_k4,
        name="SC-FNO-v2 K=4",
        q=q,
        t=t,
        bathymetry=bathymetry,
    )

    # ============================================================
    # PARAMETER SHARING CHECK
    # ============================================================

    k2_parameters = (
        k2_result[
            "parameter_count"
        ]
    )

    k4_parameters = (
        k4_result[
            "parameter_count"
        ]
    )

    original_parameters = (
        original_result[
            "parameter_count"
        ]
    )

    print(
        "\n========================================"
    )

    print(
        "PARAMETER SHARING CHECK"
    )

    print(
        "========================================"
    )

    print(
        f"Original FNO: "
        f"{original_parameters:,}"
    )

    print(
        f"SC-FNO-v2 K=2: "
        f"{k2_parameters:,}"
    )

    print(
        f"SC-FNO-v2 K=4: "
        f"{k4_parameters:,}"
    )

    if (
        k2_parameters
        !=
        k4_parameters
    ):

        raise RuntimeError(
            "K=2 and K=4 have different "
            "parameter counts. "
            "Parameter sharing failed."
        )

    print(
        "\nPASS:"
    )

    print(
        "K=2 and K=4 have identical "
        "parameter counts."
    )

    print(
        "The Fourier/local operator "
        "is genuinely shared."
    )

    reduction = (
        1.0
        -
        (
            k4_parameters
            /
            original_parameters
        )
    )

    print(
        "\nParameter reduction relative "
        "to original FNO:"
    )

    print(
        f"{100.0 * reduction:.2f}%"
    )

    print(
        "\n========================================"
    )

    print(
        "SC-FNO-v2 SMOKE TEST PASSED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()