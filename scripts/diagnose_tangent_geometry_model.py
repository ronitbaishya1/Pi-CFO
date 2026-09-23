"""
Directional tangent diagnostic for trained geometry-aware CFO models.

Supported architectures
-----------------------

    geometry_ufno
    geometry_dit

For every physical state we generate one Gaussian noise field xi and
construct two normalized perturbations:

    v_random = xi / ||xi||

and

    v_smooth = G_sigma * xi / ||G_sigma * xi||.

Therefore random and smooth measurements originate from the same
underlying random realization.

For each direction we compare

    J_model(q,b,t) v

against

    J_SWE(q,b) v

using exact JAX JVPs.

No full Jacobian is formed.
No retraining is performed.
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

import jax
import jax.numpy as jnp

import numpy as np


# =====================================================================
# PROJECT
# =====================================================================

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


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from models.geometry_dit import (
    GeometryDiT2d,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.checkpoints import (
    load_train_state,
)

from utils.tangent_swe_bathy import (
    swe_bathy_rhs_interior,
    random_and_smooth_directions,
    symmetric_normalized_tangent_error,
)

from scripts.train_wb_bathy_bed_pi_cfo import (
    trajectory_time_array,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare neural and SWE tangent "
            "responses under random and smooth "
            "state perturbations."
        )
    )

    parser.add_argument(
        "--architecture",
        required=True,
        choices=[
            "geometry_ufno",
            "geometry_dit",
        ],
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=[
            "train",
            "eval",
            "test_id",
            "test_ood",
        ],
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="best",
    )

    parser.add_argument(
        "--ckpt-step",
        type=int,
        default=None,
    )

    # -----------------------------------------------------------------
    # U-FNO
    # -----------------------------------------------------------------

    parser.add_argument(
        "--modes1",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--modes2",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-u-blocks",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # DIT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--dit-patch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--dit-hidden-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dit-depth",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dit-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dit-mlp-ratio",
        type=float,
        default=4.0,
    )

    # -----------------------------------------------------------------
    # COMMON GEOMETRY
    # -----------------------------------------------------------------

    parser.add_argument(
        "--geometry-width",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--geometry-depth",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # PHYSICS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--x-length",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--y-length",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    # -----------------------------------------------------------------
    # TANGENT SAMPLING
    # -----------------------------------------------------------------

    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num-times",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-directions",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--smooth-kernel-size",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


# =====================================================================
# BUILD MODEL
# =====================================================================

def build_model(
    args,
    *,
    num_channels,
    dx,
    dy,
):

    if (
        args.architecture
        ==
        "geometry_ufno"
    ):

        return GeometryUFNO2d(
            num_channels=num_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            width=args.width,
            num_blocks=args.num_blocks,
            num_u_blocks=(
                args.num_u_blocks
            ),
            geometry_width=(
                args.geometry_width
            ),
            geometry_depth=(
                args.geometry_depth
            ),
            dx=dx,
            dy=dy,
            use_time=True,
        )

    if (
        args.architecture
        ==
        "geometry_dit"
    ):

        return GeometryDiT2d(
            num_channels=num_channels,
            patch_size=(
                args.dit_patch_size
            ),
            hidden_size=(
                args.dit_hidden_size
            ),
            depth=(
                args.dit_depth
            ),
            num_heads=(
                args.dit_heads
            ),
            mlp_ratio=(
                args.dit_mlp_ratio
            ),
            geometry_width=(
                args.geometry_width
            ),
            geometry_depth=(
                args.geometry_depth
            ),
            dx=dx,
            dy=dy,
            use_time=True,
        )

    raise ValueError(
        args.architecture
    )


# =====================================================================
# EMPTY METRIC STORAGE
# =====================================================================

def empty_metrics():

    return {
        "relative_error": [],
        "symmetric_error": [],
        "cosine": [],
        "model_gain": [],
        "physics_gain": [],
        "gain_ratio": [],
        "model_growth": [],
        "physics_growth": [],
    }


# =====================================================================
# MEASURE ONE DIRECTION
# =====================================================================

def measure_direction(
    *,
    model_vector_field,
    physics_vector_field,
    q,
    direction,
):

    eps = 1.0e-12

    # -----------------------------------------------------------------
    # MODEL JVP
    # -----------------------------------------------------------------

    (
        _,
        model_jvp_full,
    ) = jax.jvp(
        model_vector_field,
        (
            q,
        ),
        (
            direction,
        ),
    )

    model_jvp = (
        model_jvp_full[
            :,
            1:-1,
            1:-1,
            :,
        ]
    )

    # -----------------------------------------------------------------
    # PHYSICAL SWE JVP
    # -----------------------------------------------------------------

    (
        _,
        physics_jvp,
    ) = jax.jvp(
        physics_vector_field,
        (
            q,
        ),
        (
            direction,
        ),
    )

    direction_interior = (
        direction[
            :,
            1:-1,
            1:-1,
            :,
        ]
    )

    # -----------------------------------------------------------------
    # FLATTEN
    # -----------------------------------------------------------------

    model_flat = model_jvp.reshape(
        -1
    )

    physics_flat = physics_jvp.reshape(
        -1
    )

    direction_flat = (
        direction_interior.reshape(
            -1
        )
    )

    model_norm = jnp.linalg.norm(
        model_flat
    )

    physics_norm = jnp.linalg.norm(
        physics_flat
    )

    direction_norm = jnp.linalg.norm(
        direction_flat
    )

    # -----------------------------------------------------------------
    # PHYSICS-NORMALIZED ERROR
    # -----------------------------------------------------------------

    relative_error = (
        jnp.linalg.norm(
            model_flat
            -
            physics_flat
        )
        /
        (
            physics_norm
            +
            eps
        )
    )

    # -----------------------------------------------------------------
    # SYMMETRIC NORMALIZED ERROR
    #
    # This is the actual normalized quantity used later for L_tangent.
    # -----------------------------------------------------------------

    symmetric_error = (
        symmetric_normalized_tangent_error(
            model_jvp,
            physics_jvp,
            eps=eps,
        )
    )

    # -----------------------------------------------------------------
    # DIRECTION ALIGNMENT
    # -----------------------------------------------------------------

    cosine = (
        jnp.vdot(
            model_flat,
            physics_flat,
        )
        /
        (
            model_norm
            *
            physics_norm
            +
            eps
        )
    )

    # -----------------------------------------------------------------
    # RESPONSE MAGNITUDE
    # -----------------------------------------------------------------

    model_gain = (
        model_norm
        /
        (
            direction_norm
            +
            eps
        )
    )

    physics_gain = (
        physics_norm
        /
        (
            direction_norm
            +
            eps
        )
    )

    gain_ratio = (
        model_gain
        /
        (
            physics_gain
            +
            eps
        )
    )

    # -----------------------------------------------------------------
    # PROJECTED LOCAL GROWTH
    #
    # Diagnostic only.
    # Do not interpret this alone as a stability certificate.
    # -----------------------------------------------------------------

    model_growth = (
        jnp.vdot(
            direction_flat,
            model_flat,
        )
        /
        (
            jnp.vdot(
                direction_flat,
                direction_flat,
            )
            +
            eps
        )
    )

    physics_growth = (
        jnp.vdot(
            direction_flat,
            physics_flat,
        )
        /
        (
            jnp.vdot(
                direction_flat,
                direction_flat,
            )
            +
            eps
        )
    )

    return {
        "relative_error":
            float(
                relative_error
            ),

        "symmetric_error":
            float(
                symmetric_error
            ),

        "cosine":
            float(
                cosine
            ),

        "model_gain":
            float(
                model_gain
            ),

        "physics_gain":
            float(
                physics_gain
            ),

        "gain_ratio":
            float(
                gain_ratio
            ),

        "model_growth":
            float(
                model_growth
            ),

        "physics_growth":
            float(
                physics_growth
            ),
    }


# =====================================================================
# APPEND METRICS
# =====================================================================

def append_metrics(
    storage,
    result,
):

    for key in storage:

        storage[
            key
        ].append(
            result[
                key
            ]
        )


# =====================================================================
# PRINT METRICS
# =====================================================================

def print_metrics(
    title,
    metrics,
):

    arrays = {
        key:
            np.asarray(
                value
            )
        for key, value
        in metrics.items()
    }

    print()

    print(
        "-" * 72
    )

    print(
        title
    )

    print(
        "-" * 72
    )

    print(
        "Samples:",
        arrays[
            "relative_error"
        ].size,
    )

    print()

    print(
        "Tangent relative error"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['relative_error']):.6f}",
    )

    print(
        "  median :",
        f"{np.median(arrays['relative_error']):.6f}",
    )

    print()

    print(
        "Symmetric tangent error"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['symmetric_error']):.6f}",
    )

    print(
        "  median :",
        f"{np.median(arrays['symmetric_error']):.6f}",
    )

    print()

    print(
        "Tangent cosine similarity"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['cosine']):.6f}",
    )

    print(
        "  median :",
        f"{np.median(arrays['cosine']):.6f}",
    )

    print()

    print(
        "Model tangent gain"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['model_gain']):.6f}",
    )

    print()

    print(
        "Physical tangent gain"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['physics_gain']):.6f}",
    )

    print()

    print(
        "Model / physical gain ratio"
    )

    print(
        "  mean   :",
        f"{np.mean(arrays['gain_ratio']):.6f}",
    )

    print(
        "  median :",
        f"{np.median(arrays['gain_ratio']):.6f}",
    )

    print()

    print(
        "Projected local growth"
    )

    print(
        "  model mean   :",
        f"{np.mean(arrays['model_growth']):.6f}",
    )

    print(
        "  physics mean :",
        f"{np.mean(arrays['physics_growth']):.6f}",
    )

    return arrays


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # PATHS
    # -----------------------------------------------------------------

    dataset_path = Path(
        args.dataset_path
    ).expanduser()

    if not dataset_path.is_absolute():

        dataset_path = (
            PROJECT_ROOT
            /
            dataset_path
        )

    dataset_path = (
        dataset_path.resolve()
    )

    ckpt_dir = Path(
        args.ckpt_dir
    ).expanduser()

    if not ckpt_dir.is_absolute():

        ckpt_dir = (
            PROJECT_ROOT
            /
            ckpt_dir
        )

    ckpt_dir = (
        ckpt_dir.resolve()
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    dataset = (
        load_bathymetry_dataset(
            dataset_path
        )
    )

    split = require_split(
        dataset,
        args.split,
    )

    q_all = np.asarray(
        split[
            "q"
        ],
        dtype=np.float32,
    )

    bathymetry_all = np.asarray(
        split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    time_all = trajectory_time_array(
        split
    )

    (
        total_trajectories,
        total_times,
        nx,
        ny,
        num_channels,
    ) = q_all.shape

    dx = (
        args.x_length
        /
        nx
    )

    dy = (
        args.y_length
        /
        ny
    )

    # -----------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------

    model = build_model(
        args,
        num_channels=num_channels,
        dx=dx,
        dy=dy,
    )

    method = (
        WellBalancedBathymetryBedPICFO(
            model=model,
            input_shape=(
                nx,
                ny,
                num_channels,
            ),
            condition_shape=(
                nx,
                ny,
                1,
            ),
            gamma=1.0e-5,
            spline_type="quintic",
            lambda_pde=0.03,
            lambda_bed=0.70,
            lambda_wb=0.10,
            wb_eta0=1.5,
            dx=dx,
            dy=dy,
            gravity=args.gravity,
        )
    )

    state_template = (
        init_cfo_train_state(
            method,
            seed=0,
            learning_rate=1.0e-4,
            beta1=0.9,
            beta2=0.99,
        )
    )

    state = load_train_state(
        target_state=(
            state_template
        ),
        ckpt_dir=str(
            ckpt_dir
        ),
        prefix=(
            args.ckpt_prefix
        ),
        step=(
            args.ckpt_step
        ),
    )

    # -----------------------------------------------------------------
    # STATE SAMPLE INDICES
    # -----------------------------------------------------------------

    num_trajectories = min(
        args.num_trajectories,
        total_trajectories,
    )

    trajectory_indices = np.linspace(
        0,
        total_trajectories - 1,
        num_trajectories,
        dtype=int,
    )

    if total_times > 2:

        first_time_index = 1

        last_time_index = (
            total_times
            -
            2
        )

    else:

        first_time_index = 0

        last_time_index = (
            total_times
            -
            1
        )

    num_times = min(
        args.num_times,
        (
            last_time_index
            -
            first_time_index
            +
            1
        ),
    )

    time_indices = np.linspace(
        first_time_index,
        last_time_index,
        num_times,
        dtype=int,
    )

    # -----------------------------------------------------------------
    # METRIC STORAGE
    # -----------------------------------------------------------------

    random_metrics = empty_metrics()

    smooth_metrics = empty_metrics()

    rng = jax.random.PRNGKey(
        args.seed
    )

    # -----------------------------------------------------------------
    # STATES
    # -----------------------------------------------------------------

    for trajectory_index in (
        trajectory_indices
    ):

        bathymetry = jnp.asarray(
            bathymetry_all[
                trajectory_index:
                trajectory_index + 1
            ]
        )

        for time_index in (
            time_indices
        ):

            q = jnp.asarray(
                q_all[
                    trajectory_index:
                    trajectory_index + 1,
                    time_index,
                ]
            )

            time_value = float(
                time_all[
                    trajectory_index,
                    time_index,
                ]
                if time_all.ndim == 2
                else
                time_all[
                    time_index
                ]
            )

            physical_time = jnp.asarray(
                [
                    time_value
                ],
                dtype=jnp.float32,
            )

            # -----------------------------------------------------
            # MODEL FIELD AT FIXED b,t
            # -----------------------------------------------------

            def model_vector_field(
                q_input,
            ):

                return model.apply(
                    {
                        "params":
                            state.params
                    },
                    q_input,
                    physical_time,
                    bathymetry,
                )

            # -----------------------------------------------------
            # PHYSICAL SWE FIELD AT FIXED b
            # -----------------------------------------------------

            def physics_vector_field(
                q_input,
            ):

                return (
                    swe_bathy_rhs_interior(
                        q_input,
                        bathymetry,
                        dx=dx,
                        dy=dy,
                        g=args.gravity,
                    )
                )

            # -----------------------------------------------------
            # PAIRED RANDOM/SMOOTH DIRECTIONS
            # -----------------------------------------------------

            for _ in range(
                args.num_directions
            ):

                (
                    rng,
                    noise_key,
                ) = jax.random.split(
                    rng
                )

                raw_noise = (
                    jax.random.normal(
                        noise_key,
                        q.shape,
                        dtype=q.dtype,
                    )
                )

                (
                    random_direction,
                    smooth_direction,
                ) = random_and_smooth_directions(
                    raw_noise,
                    sigma=(
                        args.smooth_sigma
                    ),
                    kernel_size=(
                        args.smooth_kernel_size
                    ),
                )

                # -------------------------------------------------
                # RANDOM
                # -------------------------------------------------

                random_result = (
                    measure_direction(
                        model_vector_field=(
                            model_vector_field
                        ),
                        physics_vector_field=(
                            physics_vector_field
                        ),
                        q=q,
                        direction=(
                            random_direction
                        ),
                    )
                )

                append_metrics(
                    random_metrics,
                    random_result,
                )

                # -------------------------------------------------
                # SMOOTH
                # -------------------------------------------------

                smooth_result = (
                    measure_direction(
                        model_vector_field=(
                            model_vector_field
                        ),
                        physics_vector_field=(
                            physics_vector_field
                        ),
                        q=q,
                        direction=(
                            smooth_direction
                        ),
                    )
                )

                append_metrics(
                    smooth_metrics,
                    smooth_result,
                )

    # -----------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "RANDOM VS SMOOTH TANGENT DIAGNOSTIC"
    )

    print(
        "=" * 72
    )

    print(
        "Architecture:",
        args.architecture,
    )

    print(
        "Resolution:",
        f"{nx} x {ny}",
    )

    print(
        "Smooth sigma:",
        args.smooth_sigma,
        "cells",
    )

    print(
        "Smooth kernel:",
        (
            f"{args.smooth_kernel_size}"
            " x "
            f"{args.smooth_kernel_size}"
        ),
    )

    random_arrays = print_metrics(
        "RANDOM PERTURBATIONS",
        random_metrics,
    )

    smooth_arrays = print_metrics(
        "SMOOTH PERTURBATIONS",
        smooth_metrics,
    )

    # -----------------------------------------------------------------
    # SIMPLE SUMMARY
    # -----------------------------------------------------------------

    print()

    print(
        "=" * 72
    )

    print(
        "SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "Random cosine:",
        f"{np.mean(random_arrays['cosine']):.6f}",
    )

    print(
        "Smooth cosine:",
        f"{np.mean(smooth_arrays['cosine']):.6f}",
    )

    print(
        "Random gain ratio:",
        f"{np.mean(random_arrays['gain_ratio']):.6f}",
    )

    print(
        "Smooth gain ratio:",
        f"{np.mean(smooth_arrays['gain_ratio']):.6f}",
    )

    print(
        "Random symmetric error:",
        f"{np.mean(random_arrays['symmetric_error']):.6f}",
    )

    print(
        "Smooth symmetric error:",
        f"{np.mean(smooth_arrays['symmetric_error']):.6f}",
    )

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    if args.output is None:

        output_path = (
            PROJECT_ROOT
            /
            "results"
            /
            (
                "tangent_random_smooth_"
                +
                args.architecture
                +
                ".npz"
            )
        )

    else:

        output_path = (
            Path(
                args.output
            )
            .expanduser()
        )

        if not output_path.is_absolute():

            output_path = (
                PROJECT_ROOT
                /
                output_path
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dictionary = {
        "architecture":
            np.asarray(
                args.architecture
            ),

        "smooth_sigma":
            np.asarray(
                args.smooth_sigma
            ),

        "smooth_kernel_size":
            np.asarray(
                args.smooth_kernel_size
            ),
    }

    for key, value in (
        random_arrays.items()
    ):

        save_dictionary[
            "random_"
            +
            key
        ] = value

    for key, value in (
        smooth_arrays.items()
    ):

        save_dictionary[
            "smooth_"
            +
            key
        ] = value

    np.savez(
        output_path,
        **save_dictionary,
    )

    print()

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":

    main()