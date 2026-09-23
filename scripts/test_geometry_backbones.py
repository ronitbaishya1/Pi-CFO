"""
Smoke-test geometry-conditioned operator backbones at 32x32 and 64x64.

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


from models.geometry_fno import (
    GeometryFNO2d,
)

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from models.geometry_ffno import (
    GeometryFFNO2d,
)

from models.geometry_transolver import (
    GeometryTransolver2d,
)

from models.geometry_dit import (
    GeometryDiT2d,
)


def parameter_count(
    params,
):

    return sum(
        leaf.size
        for leaf
        in tree_leaves(
            params
        )
    )


def build_models(
    *,
    dx,
    dy,
):

    return {

        "geometry_fno":
            GeometryFNO2d(
                num_channels=3,
                modes1=8,
                modes2=8,
                width=32,
                num_blocks=4,
                geometry_width=16,
                geometry_depth=2,
                dx=dx,
                dy=dy,
                use_time=True,
            ),

        "geometry_ufno":
            GeometryUFNO2d(
                num_channels=3,
                modes1=8,
                modes2=8,
                width=32,
                num_blocks=4,
                num_u_blocks=2,
                geometry_width=16,
                geometry_depth=2,
                dx=dx,
                dy=dy,
                use_time=True,
            ),

        "geometry_ffno":
            GeometryFFNO2d(
                num_channels=3,
                modes1=8,
                modes2=8,
                width=32,
                num_blocks=4,
                expansion=2,
                geometry_width=16,
                geometry_depth=2,
                dx=dx,
                dy=dy,
                use_time=True,
            ),

        "geometry_transolver":
            GeometryTransolver2d(
                num_channels=3,
                width=32,
                num_blocks=4,
                num_heads=4,
                slice_num=16,
                mlp_ratio=2,
                geometry_width=16,
                geometry_depth=2,
                dx=dx,
                dy=dy,
                use_time=True,
            ),

        "geometry_dit":
            GeometryDiT2d(
                num_channels=3,
                patch_size=4,
                hidden_size=256,
                depth=4,
                num_heads=4,
                mlp_ratio=4.0,
                geometry_width=16,
                geometry_depth=2,
                dx=dx,
                dy=dy,
                use_time=True,
            ),
    }


def run_resolution_test(
    resolution,
):

    print()

    print(
        "=" * 70
    )

    print(
        f"Testing resolution "
        f"{resolution} x {resolution}"
    )

    print(
        "=" * 70
    )

    domain_length = 5.0

    dx = (
        domain_length
        /
        resolution
    )

    dy = (
        domain_length
        /
        resolution
    )

    batch_size = 2

    # -------------------------------------------------------------
    # SWE state q = [h,hu,hv]
    # -------------------------------------------------------------

    x = jnp.ones(
        (
            batch_size,
            resolution,
            resolution,
            3,
        ),
        dtype=jnp.float32,
    )

    # -------------------------------------------------------------
    # Bathymetry
    # -------------------------------------------------------------

    geometry = jnp.zeros(
        (
            batch_size,
            resolution,
            resolution,
            1,
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
        0.3
        *
        jnp.exp(
            -(
                xx**2
                +
                yy**2
            )
            /
            (
                2.0
                *
                0.5**2
            )
        )
    )

    geometry = geometry.at[
        :,
        :,
        :,
        0,
    ].set(
        hill[
            None,
            ...
        ]
    )

    t = jnp.asarray(
        [
            0.25,
            0.75,
        ],
        dtype=jnp.float32,
    )

    models = build_models(
        dx=dx,
        dy=dy,
    )

    for index, (
        name,
        model,
    ) in enumerate(
        models.items()
    ):

        print()

        print(
            f"Testing {name}..."
        )

        key = jax.random.PRNGKey(
            100
            +
            index
            +
            resolution
        )

        variables = model.init(
            key,
            x,
            t,
            geometry,
        )

        output = model.apply(
            variables,
            x,
            t,
            geometry,
        )

        if output.shape != x.shape:

            raise RuntimeError(
                f"{name}: output shape "
                f"{output.shape} does not "
                f"match state shape {x.shape}."
            )

        if not bool(
            jnp.all(
                jnp.isfinite(
                    output
                )
            )
        ):

            raise RuntimeError(
                f"{name}: output contains "
                "NaN or Inf."
            )

        count = parameter_count(
            variables[
                "params"
            ]
        )

        print(
            f"{name:22s} "
            f"shape={str(output.shape):20s} "
            f"params={count:,}"
        )


def main():

    run_resolution_test(
        32
    )

    run_resolution_test(
        64
    )

    print()

    print(
        "=" * 70
    )

    print(
        "All geometry-backbone "
        "smoke tests passed."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()