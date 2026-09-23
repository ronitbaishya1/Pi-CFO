"""
Smoke test for GeometryTransolver2d.

Tests:
    - 32x32 forward pass
    - 64x64 forward pass
    - output shape
    - finite output
    - JIT compatibility
    - parameter count

No training is performed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from jax.tree_util import tree_leaves


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


from models.geometry_transolver import (
    GeometryTransolver2d,
)


# =====================================================================
# PARAMETER COUNT
# =====================================================================

def parameter_count(
    params,
):

    return sum(
        leaf.size
        for leaf in tree_leaves(
            params
        )
    )


# =====================================================================
# TEST ONE RESOLUTION
# =====================================================================

def test_resolution(
    resolution,
):

    print()

    print(
        "=" * 72
    )

    print(
        f"Testing GeometryTransolver2d "
        f"at {resolution}x{resolution}"
    )

    print(
        "=" * 72
    )

    dx = (
        5.0
        /
        resolution
    )

    dy = (
        5.0
        /
        resolution
    )

    batch_size = 2

    state = jnp.ones(
        (
            batch_size,
            resolution,
            resolution,
            3,
        ),
        dtype=jnp.float32,
    )

    coord = jnp.linspace(
        -2.5,
        2.5,
        resolution,
    )

    xx, yy = jnp.meshgrid(
        coord,
        coord,
        indexing="ij",
    )

    hill = (
        0.20
        *
        jnp.exp(
            -0.5
            *
            (
                xx**2
                +
                yy**2
            )
            /
            (
                0.5**2
            )
        )
    )

    geometry = jnp.tile(
        hill[
            None,
            ...,
            None,
        ],
        (
            batch_size,
            1,
            1,
            1,
        ),
    )

    time = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    model = GeometryTransolver2d(
        num_channels=3,

        dim=64,

        num_blocks=4,

        heads=4,

        dim_head=16,

        slice_num=32,

        ff_multiplier=2,

        kernel_size=3,

        geometry_width=16,

        geometry_depth=2,

        dx=dx,

        dy=dy,

        use_time=True,
    )

    key = jax.random.PRNGKey(
        resolution
    )

    variables = model.init(
        key,
        state,
        time,
        geometry,
    )

    output = model.apply(
        variables,
        state,
        time,
        geometry,
    )

    expected_shape = (
        state.shape
    )

    if output.shape != expected_shape:

        raise RuntimeError(
            f"Wrong output shape: "
            f"{output.shape}, "
            f"expected {expected_shape}"
        )

    if not bool(
        jnp.all(
            jnp.isfinite(
                output
            )
        )
    ):

        raise RuntimeError(
            "Output contains NaN or Inf."
        )

    # =============================================================
    # JIT TEST
    # =============================================================

    apply_jit = jax.jit(
        lambda variables, x, t, c:
            model.apply(
                variables,
                x,
                t,
                c,
            )
    )

    output_jit = apply_jit(
        variables,
        state,
        time,
        geometry,
    )

    if output_jit.shape != expected_shape:

        raise RuntimeError(
            "JIT output has wrong shape."
        )

    count = parameter_count(
        variables[
            "params"
        ]
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Parameters:",
        f"{count:,}",
    )

    print(
        "Finite output:",
        True,
    )

    print(
        "JIT:",
        "passed",
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    test_resolution(
        32
    )

    test_resolution(
        64
    )

    print()

    print(
        "=" * 72
    )

    print(
        "ALL TRANSOLVER SMOKE TESTS PASSED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()