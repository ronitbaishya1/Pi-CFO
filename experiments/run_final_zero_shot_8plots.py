"""
FINAL 64 -> 128 ZERO-SHOT RESOLUTION STUDY
==========================================

Compare:

    1. PyClaw 64x64
    2. Bathy-CFO trained at 64 -> evaluated zero-shot at 128
    3. Final Geometry-U-FNO trained at 64 -> evaluated zero-shot at 128
       using:
           canonical U-Net resolution = 64
           128 -> 64 = area averaging
           U-Net at 64
           64 -> 128 = cubic reconstruction
           native blend = 0
           canonical geometry = OFF
           spectral low-pass = OFF
    4. PyClaw 128x128

No 128x128 neural-network training is performed.

The canonical U-Net operates INSIDE every model evaluation, therefore
it is also active during every RK4 stage of the zero-shot rollout.

Outputs
-------
01_direct_condition_test.png
02_isolated_hill_effect_error.png
03_terrain_induced_speed_gaussian.png
04_terrain_induced_speed_unseen.png
05_gaussian_cross_sections_h_hu_hv.png
06_gaussian_3d_bathymetry_cross_sections.png
07_gaussian_3d_free_surface.png
08_gaussian_3d_h_hu_hv_cross_sections.png

zero_shot_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
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

from cfo import (
    ContinuousFlowOperator,
)

from models.factory import (
    build_model,
)

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)


# =====================================================================
# CONSTANTS
# =====================================================================

SOURCE_RESOLUTION = 64
TARGET_RESOLUTION = 128

X_MIN = -2.5
X_MAX = 2.5

Y_MIN = -2.5
Y_MAX = 2.5

LX = X_MAX - X_MIN
LY = Y_MAX - Y_MIN

DX64 = LX / SOURCE_RESOLUTION
DY64 = LY / SOURCE_RESOLUTION

DX128 = LX / TARGET_RESOLUTION
DY128 = LY / TARGET_RESOLUTION

GRAVITY = 1.0


GAUSSIAN_CASES = [
    "hill_left",
    "hill_center",
    "hill_right",
]


OOD_CASES = [
    "two_hills",
    "narrow_tall",
    "elongated_ridge",
    "rotated_ridge",
    "multi_hill",
]


METHOD_LABELS = {
    "pyclaw64":
        "PyClaw 64×64",

    "cfo":
        "Bathy-CFO 64→128",

    "final":
        "Canonical Geo-U-FNO 64→128",

    "pyclaw128":
        "PyClaw 128×128",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gaussian64",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--gaussian128",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_128.h5"
        ),
    )

    parser.add_argument(
        "--ood64",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_64.h5"
        ),
    )

    parser.add_argument(
        "--ood128",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_128.h5"
        ),
    )

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/final_resolution_study/"
            "bathy_cfo/res64/seed0/best"
        ),
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    parser.add_argument(
        "--final-ckpt",
        type=str,
        default=(
            "checkpoints/rollout_stable/"
            "geometry_ufno/res64/"
            "K2_lamroll_0p1_lamhf_0/"
            "seed0/best"
        ),
    )

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lambda-bed",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--lambda-wb",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--direct-case",
        type=str,
        default="hill_right",
        choices=GAUSSIAN_CASES,
    )

    parser.add_argument(
        "--cross-section-case",
        type=str,
        default="hill_center",
        choices=GAUSSIAN_CASES,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/final_project/"
            "zero_shot_64_to_128"
        ),
    )

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve_path(value):

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# HDF5 CASE RESOLUTION
# =====================================================================

def find_case_name(
    cases_group,
    requested,
):

    if requested in cases_group:

        return requested

    candidates = [
        name
        for name in cases_group.keys()
        if name.startswith(
            requested + "_"
        )
    ]

    if len(candidates) > 0:

        return sorted(
            candidates
        )[0]

    raise KeyError(
        f"Could not find case '{requested}'.\n"
        f"Available cases:\n"
        f"{list(cases_group.keys())}"
    )


# =====================================================================
# LOAD ONE CASE
# =====================================================================

def load_case(
    file_path,
    case_name,
):

    file_path = resolve_path(
        file_path
    )

    with h5py.File(
        file_path,
        "r",
    ) as h5:

        if "cases" not in h5:

            raise RuntimeError(
                f"{file_path} does not contain "
                "a 'cases' group."
            )

        actual_name = find_case_name(
            h5["cases"],
            case_name,
        )

        group = (
            h5[
                "cases"
            ][
                actual_name
            ]
        )

        q = np.asarray(
            group[
                "q"
            ],
            dtype=np.float32,
        )

        b = np.asarray(
            group[
                "bathymetry"
            ],
            dtype=np.float32,
        )

        time = np.asarray(
            h5[
                "time"
            ],
            dtype=np.float32,
        )

        x = np.asarray(
            h5[
                "x"
            ],
            dtype=np.float32,
        )

        y = np.asarray(
            h5[
                "y"
            ],
            dtype=np.float32,
        )

    # -------------------------------------------------------------
    # Normalize q
    # -------------------------------------------------------------

    while q.ndim > 4:

        q = q[
            0
        ]

    if (
        q.ndim != 4
        or
        q.shape[-1] != 3
    ):

        raise ValueError(
            f"Unexpected q shape: {q.shape}"
        )

    # -------------------------------------------------------------
    # Normalize bathymetry
    # -------------------------------------------------------------

    while b.ndim > 3:

        b = b[
            0
        ]

    if (
        b.ndim == 3
        and
        b.shape[-1] == 1
    ):

        b = b[
            ...,
            0
        ]

    if b.ndim != 2:

        raise ValueError(
            f"Unexpected bathymetry shape: "
            f"{b.shape}"
        )

    return {
        "name":
            actual_name,

        "q":
            q,

        "b":
            b,

        "time":
            time,

        "x":
            x,

        "y":
            y,
    }


# =====================================================================
# LOAD TERRAIN + FLAT PAIR
# =====================================================================

def load_pair(
    file_path,
    case_name,
):

    terrain = load_case(
        file_path,
        case_name,
    )

    flat = load_case(
        file_path,
        "flat",
    )

    if (
        terrain[
            "q"
        ].shape
        !=
        flat[
            "q"
        ].shape
    ):

        raise ValueError(
            "Terrain and flat trajectories "
            "have different shapes."
        )

    return {
        "terrain":
            terrain,

        "flat":
            flat,
    }


# =====================================================================
# INTERPOLATION 64 -> 128
# =====================================================================

def interp2(
    field,
    x_source,
    y_source,
    x_target,
    y_target,
):

    temporary = np.empty(
        (
            len(
                x_target
            ),
            len(
                y_source
            ),
        ),
        dtype=np.float64,
    )

    for j in range(
        len(
            y_source
        )
    ):

        temporary[
            :,
            j
        ] = np.interp(
            x_target,
            x_source,
            field[
                :,
                j
            ],
        )

    output = np.empty(
        (
            len(
                x_target
            ),
            len(
                y_target
            ),
        ),
        dtype=np.float64,
    )

    for i in range(
        len(
            x_target
        )
    ):

        output[
            i,
            :
        ] = np.interp(
            y_target,
            y_source,
            temporary[
                i,
                :
            ],
        )

    return output


def interpolate_state(
    q,
    x_source,
    y_source,
    x_target,
    y_target,
):

    nt = q.shape[
        0
    ]

    nc = q.shape[
        -1
    ]

    output = np.empty(
        (
            nt,
            len(
                x_target
            ),
            len(
                y_target
            ),
            nc,
        ),
        dtype=np.float32,
    )

    for time_index in range(
        nt
    ):

        for channel in range(
            nc
        ):

            output[
                time_index,
                ...,
                channel
            ] = interp2(
                q[
                    time_index,
                    ...,
                    channel
                ],

                x_source,

                y_source,

                x_target,

                y_target,
            )

    return output


# =====================================================================
# BUILD BATHY-CFO AT TARGET GRID
# =====================================================================

def build_cfo_method():

    input_shape = (
        TARGET_RESOLUTION,
        TARGET_RESOLUTION,
        3,
    )

    condition_shape = (
        TARGET_RESOLUTION,
        TARGET_RESOLUTION,
        1,
    )

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return ContinuousFlowOperator(
        model=model,

        input_shape=input_shape,

        gamma=1.0e-5,

        spline_type="quintic",

        use_condition=True,

        condition_shape=(
            condition_shape
        ),
    )


# =====================================================================
# BUILD FINAL CANONICAL GEO-U-FNO
# =====================================================================

def build_final_method(
    args,
):

    model = GeometryUFNO2d(
        num_channels=3,

        modes1=12,
        modes2=12,

        width=64,

        num_blocks=4,
        num_u_blocks=2,

        geometry_width=16,
        geometry_depth=2,

        # ---------------------------------------------------------
        # TARGET GRID SPACING
        # ---------------------------------------------------------

        dx=DX128,
        dy=DY128,

        include_gradient_magnitude=False,

        use_time=True,

        # ---------------------------------------------------------
        # FINAL ZERO-SHOT CANONICAL CONFIGURATION
        # ---------------------------------------------------------

        canonical_unet_resolution=64,

        canonical_geometry_resolution=None,

        canonical_downsample_mode="area",

        canonical_upsample_mode="cubic",

        canonical_native_blend=0.0,

        source_resolution_filter=None,

        domain_length_x=LX,

        domain_length_y=LY,
    )

    return WellBalancedBathymetryBedPICFO(
        model=model,

        input_shape=(
            TARGET_RESOLUTION,
            TARGET_RESOLUTION,
            3,
        ),

        condition_shape=(
            TARGET_RESOLUTION,
            TARGET_RESOLUTION,
            1,
        ),

        gamma=1.0e-5,

        spline_type="quintic",

        lambda_pde=(
            args.lambda_pde
        ),

        lambda_bed=(
            args.lambda_bed
        ),

        lambda_wb=(
            args.lambda_wb
        ),

        wb_eta0=(
            args.wb_eta0
        ),

        dx=DX128,

        dy=DY128,

        gravity=GRAVITY,
    )


# =====================================================================
# RESTORE BATHY-CFO
# =====================================================================

def restore_cfo_state(
    method,
    args,
):

    target = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,

        beta2=0.99,
    )

    return load_train_state(
        target,

        ckpt_dir=str(
            resolve_path(
                args.cfo_ckpt_dir
            )
        ),

        prefix=(
            args.cfo_prefix
        ),

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# RESTORE FINAL GEO-U-FNO
# =====================================================================

def restore_final_state(
    method,
    args,
):

    checkpoint = resolve_path(
        args.final_ckpt
    )

    target = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,

        beta2=0.99,
    )

    return load_train_state(
        target,

        ckpt_dir=str(
            checkpoint.parent
        ),

        prefix=(
            checkpoint.name
        ),

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# NORMALIZE MODEL PREDICTION
# =====================================================================

def normalize_prediction(
    prediction,
    expected_times,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if (
        prediction.ndim == 5
        and
        prediction.shape[0] == expected_times
        and
        prediction.shape[1] == 1
    ):

        return prediction[
            :,
            0
        ]

    if (
        prediction.ndim == 5
        and
        prediction.shape[0] == 1
        and
        prediction.shape[1] == expected_times
    ):

        return prediction[
            0
        ]

    if prediction.ndim == 4:

        return prediction

    raise ValueError(
        "Unexpected prediction shape: "
        f"{prediction.shape}"
    )


# =====================================================================
# MODEL ROLLOUT
# =====================================================================

def model_rollout(
    method,
    state,
    q_reference,
    bathymetry,
    *,
    steps_per_segment,
):

    q0 = jnp.asarray(
        q_reference[
            0:
            1
        ],
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        bathymetry[
            None,
            ...,
            None
        ],
        dtype=jnp.float32,
    )

    prediction = method.uniform_inference(
        state,

        q0,

        trajectory_points_num=(
            q_reference.shape[
                0
            ]
        ),

        steps_per_segment=(
            steps_per_segment
        ),

        condition=condition,

        method="RK4",
    )

    return normalize_prediction(
        prediction,
        q_reference.shape[
            0
        ],
    )


# =====================================================================
# RUN ONE CASE
# =====================================================================

def evaluate_case(
    data64,
    data128,
    *,
    cfo_method,
    cfo_state,
    final_method,
    final_state,
    steps_per_segment,
):

    terrain128 = data128[
        "terrain"
    ]

    flat128 = data128[
        "flat"
    ]

    print()
    print(
        "Running Bathy-CFO on",
        terrain128[
            "name"
        ],
    )

    cfo_terrain = model_rollout(
        cfo_method,
        cfo_state,

        terrain128[
            "q"
        ],

        terrain128[
            "b"
        ],

        steps_per_segment=(
            steps_per_segment
        ),
    )

    cfo_flat = model_rollout(
        cfo_method,
        cfo_state,

        flat128[
            "q"
        ],

        flat128[
            "b"
        ],

        steps_per_segment=(
            steps_per_segment
        ),
    )

    print(
        "Running canonical Geo-U-FNO on",
        terrain128[
            "name"
        ],
    )

    final_terrain = model_rollout(
        final_method,
        final_state,

        terrain128[
            "q"
        ],

        terrain128[
            "b"
        ],

        steps_per_segment=(
            steps_per_segment
        ),
    )

    final_flat = model_rollout(
        final_method,
        final_state,

        flat128[
            "q"
        ],

        flat128[
            "b"
        ],

        steps_per_segment=(
            steps_per_segment
        ),
    )

    return {
        "data64":
            data64,

        "data128":
            data128,

        "cfo": {
            "terrain":
                cfo_terrain,

            "flat":
                cfo_flat,
        },

        "final": {
            "terrain":
                final_terrain,

            "flat":
                final_flat,
        },
    }


# =====================================================================
# BASIC METRICS
# =====================================================================

def relative_l2(
    truth,
    prediction,
    eps=1.0e-12,
):

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    numerator = np.linalg.norm(
        (
            prediction
            -
            truth
        ).ravel()
    )

    denominator = np.linalg.norm(
        truth.ravel()
    )

    return float(
        numerator
        /
        max(
            denominator,
            eps,
        )
    )


def rmse(
    truth,
    prediction,
):

    return float(
        np.sqrt(
            np.mean(
                (
                    np.asarray(
                        prediction,
                        dtype=np.float64,
                    )
                    -
                    np.asarray(
                        truth,
                        dtype=np.float64,
                    )
                )
                ** 2
            )
        )
    )


def relative_error_vs_time(
    truth,
    prediction,
):

    output = []

    for index in range(
        truth.shape[
            0
        ]
    ):

        denominator = np.linalg.norm(
            truth[
                index
            ].ravel()
        )

        numerator = np.linalg.norm(
            (
                prediction[
                    index
                ]
                -
                truth[
                    index
                ]
            ).ravel()
        )

        if denominator < 1.0e-12:

            output.append(
                0.0
                if numerator < 1.0e-12
                else numerator
            )

        else:

            output.append(
                numerator
                /
                denominator
            )

    return np.asarray(
        output,
        dtype=np.float64,
    )


# =====================================================================
# SPEED
# =====================================================================

def speed(
    q,
):

    h = np.maximum(
        q[
            ...,
            0
        ],
        1.0e-6,
    )

    u = (
        q[
            ...,
            1
        ]
        /
        h
    )

    v = (
        q[
            ...,
            2
        ]
        /
        h
    )

    return np.sqrt(
        u**2
        +
        v**2
    )


# =====================================================================
# MODEL VECTOR FIELD
# =====================================================================

def model_vector_field(
    method,
    state,
    q,
    b,
    time_value,
):

    result = method._model_apply(
        state.params,

        jnp.asarray(
            q[
                None,
                ...
            ],
            dtype=jnp.float32,
        ),

        jnp.asarray(
            [
                time_value
            ],
            dtype=jnp.float32,
        ),

        jnp.asarray(
            b[
                None,
                ...,
                None
            ],
            dtype=jnp.float32,
        ),
    )

    return np.asarray(
        result[
            0
        ],
        dtype=np.float32,
    )


# =====================================================================
# SWE BATHYMETRY SOURCE RESPONSE
# =====================================================================

def swe_bathymetry_response(
    q,
    b,
    dx,
    dy,
):

    h = np.maximum(
        q[
            ...,
            0
        ],
        1.0e-6,
    )

    bx = np.gradient(
        b,
        dx,
        axis=0,
        edge_order=2,
    )

    by = np.gradient(
        b,
        dy,
        axis=1,
        edge_order=2,
    )

    hu = (
        -GRAVITY
        *
        h
        *
        bx
    )

    hv = (
        -GRAVITY
        *
        h
        *
        by
    )

    return (
        hu,
        hv,
    )


# =====================================================================
# HEATMAP
# =====================================================================

def heatmap(
    ax,
    field,
    *,
    title,
    vlim,
    b=None,
):

    image = ax.imshow(
        field.T,

        origin="lower",

        extent=[
            X_MIN,
            X_MAX,
            Y_MIN,
            Y_MAX,
        ],

        cmap="coolwarm",

        vmin=-vlim,

        vmax=vlim,

        interpolation="bilinear",

        aspect="equal",
    )

    if b is not None:

        nx, ny = b.shape

        x = (
            X_MIN
            +
            (
                np.arange(
                    nx
                )
                +
                0.5
            )
            *
            (
                LX
                /
                nx
            )
        )

        y = (
            Y_MIN
            +
            (
                np.arange(
                    ny
                )
                +
                0.5
            )
            *
            (
                LY
                /
                ny
            )
        )

        ax.contour(
            x,
            y,
            b.T,

            levels=5,

            linewidths=0.5,

            alpha=0.45,
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "y"
    )

    return image


# =====================================================================
# FIGURE 01
# DIRECT CONDITION RESPONSE
# =====================================================================

def plot_direct_condition(
    result,
    *,
    cfo_method,
    cfo_state,
    final_method,
    final_state,
    plot_time,
    output_path,
):

    terrain64 = result[
        "data64"
    ][
        "terrain"
    ]

    terrain128 = result[
        "data128"
    ][
        "terrain"
    ]

    index64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    index128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    t128 = float(
        terrain128[
            "time"
        ][
            index128
        ]
    )

    # -------------------------------------------------------------
    # PyClaw source response at both grids
    # -------------------------------------------------------------

    hu64, hv64 = (
        swe_bathymetry_response(
            terrain64[
                "q"
            ][
                index64
            ],

            terrain64[
                "b"
            ],

            DX64,

            DY64,
        )
    )

    hu128, hv128 = (
        swe_bathymetry_response(
            terrain128[
                "q"
            ][
                index128
            ],

            terrain128[
                "b"
            ],

            DX128,

            DY128,
        )
    )

    q128 = (
        terrain128[
            "q"
        ][
            index128
        ]
    )

    b128 = (
        terrain128[
            "b"
        ]
    )

    zero_bed = np.zeros_like(
        b128
    )

    # -------------------------------------------------------------
    # Bathy-CFO response
    # -------------------------------------------------------------

    cfo_with = model_vector_field(
        cfo_method,
        cfo_state,
        q128,
        b128,
        t128,
    )

    cfo_without = model_vector_field(
        cfo_method,
        cfo_state,
        q128,
        zero_bed,
        t128,
    )

    cfo_response = (
        cfo_with
        -
        cfo_without
    )

    # -------------------------------------------------------------
    # Canonical Geo-U-FNO response
    # -------------------------------------------------------------

    final_with = model_vector_field(
        final_method,
        final_state,
        q128,
        b128,
        t128,
    )

    final_without = model_vector_field(
        final_method,
        final_state,
        q128,
        zero_bed,
        t128,
    )

    final_response = (
        final_with
        -
        final_without
    )

    hu_fields = [
        hu64,
        cfo_response[
            ...,
            1
        ],
        final_response[
            ...,
            1
        ],
        hu128,
    ]

    hv_fields = [
        hv64,
        cfo_response[
            ...,
            2
        ],
        final_response[
            ...,
            2
        ],
        hv128,
    ]

    labels = [
        METHOD_LABELS[
            "pyclaw64"
        ],

        METHOD_LABELS[
            "cfo"
        ],

        METHOD_LABELS[
            "final"
        ],

        METHOD_LABELS[
            "pyclaw128"
        ],
    ]

    beds = [
        terrain64[
            "b"
        ],

        b128,

        b128,

        b128,
    ]

    hu_limit = max(
        max(
            float(
                np.max(
                    np.abs(
                        value
                    )
                )
            )
            for value
            in hu_fields
        ),
        1.0e-10,
    )

    hv_limit = max(
        max(
            float(
                np.max(
                    np.abs(
                        value
                    )
                )
            )
            for value
            in hv_fields
        ),
        1.0e-10,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(
            20,
            9,
        ),
    )

    for column in range(
        4
    ):

        image = heatmap(
            axes[
                0,
                column
            ],

            hu_fields[
                column
            ],

            title=(
                labels[
                    column
                ]
                +
                "\nBathymetry response: hu"
            ),

            vlim=hu_limit,

            b=beds[
                column
            ],
        )

        fig.colorbar(
            image,

            ax=axes[
                0,
                column
            ],

            shrink=0.8,
        )

        image = heatmap(
            axes[
                1,
                column
            ],

            hv_fields[
                column
            ],

            title=(
                labels[
                    column
                ]
                +
                "\nBathymetry response: hv"
            ),

            vlim=hv_limit,

            b=beds[
                column
            ],
        )

        fig.colorbar(
            image,

            ax=axes[
                1,
                column
            ],

            shrink=0.8,
        )

    fig.suptitle(
        (
            "Zero-shot direct bathymetry-condition comparison "
            f"at t={t128:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 02
# ISOLATED GAUSSIAN TERRAIN EFFECT
# =====================================================================

def plot_isolated_hill_effect(
    gaussian_results,
    *,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
    )

    for column, case_name in enumerate(
        GAUSSIAN_CASES
    ):

        result = gaussian_results[
            case_name
        ]

        data64 = result[
            "data64"
        ]

        data128 = result[
            "data128"
        ]

        terrain64 = data64[
            "terrain"
        ]

        flat64 = data64[
            "flat"
        ]

        terrain128 = data128[
            "terrain"
        ]

        flat128 = data128[
            "flat"
        ]

        true_effect = (
            terrain128[
                "q"
            ]
            -
            flat128[
                "q"
            ]
        )

        q64_terrain_interp = (
            interpolate_state(
                terrain64[
                    "q"
                ],

                terrain64[
                    "x"
                ],

                terrain64[
                    "y"
                ],

                terrain128[
                    "x"
                ],

                terrain128[
                    "y"
                ],
            )
        )

        q64_flat_interp = (
            interpolate_state(
                flat64[
                    "q"
                ],

                flat64[
                    "x"
                ],

                flat64[
                    "y"
                ],

                terrain128[
                    "x"
                ],

                terrain128[
                    "y"
                ],
            )
        )

        effect64 = (
            q64_terrain_interp
            -
            q64_flat_interp
        )

        cfo_effect = (
            result[
                "cfo"
            ][
                "terrain"
            ]
            -
            result[
                "cfo"
            ][
                "flat"
            ]
        )

        final_effect = (
            result[
                "final"
            ][
                "terrain"
            ]
            -
            result[
                "final"
            ][
                "flat"
            ]
        )

        time = terrain128[
            "time"
        ]

        axes[
            column
        ].plot(
            time,

            np.zeros_like(
                time
            ),

            linestyle="--",

            label="PyClaw128 reference",
        )

        axes[
            column
        ].plot(
            time,

            relative_error_vs_time(
                true_effect,
                effect64,
            ),

            label="PyClaw64 → PyClaw128",
        )

        axes[
            column
        ].plot(
            time,

            relative_error_vs_time(
                true_effect,
                cfo_effect,
            ),

            label=METHOD_LABELS[
                "cfo"
            ],
        )

        axes[
            column
        ].plot(
            time,

            relative_error_vs_time(
                true_effect,
                final_effect,
            ),

            label=METHOD_LABELS[
                "final"
            ],
        )

        axes[
            column
        ].set_title(
            case_name
        )

        axes[
            column
        ].set_xlabel(
            "Time"
        )

        axes[
            column
        ].set_ylabel(
            "Relative terrain-effect error"
        )

        axes[
            column
        ].grid(
            alpha=0.25
        )

    axes[
        0
    ].legend(
        fontsize=8
    )

    fig.suptitle(
        "Isolated Gaussian-hill effect error",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# TERRAIN-SPEED FIELD
# =====================================================================

def terrain_speed_field(
    terrain_q,
    flat_q,
    time_index,
):

    return (
        speed(
            terrain_q[
                time_index
            ]
        )
        -
        speed(
            flat_q[
                time_index
            ]
        )
    )


# =====================================================================
# FIGURE 03 / 04
# SPEED GRID
# =====================================================================

def plot_speed_grid(
    results,
    case_names,
    *,
    plot_time,
    output_path,
    title,
):

    num_rows = len(
        case_names
    )

    fig, axes = plt.subplots(
        num_rows,
        4,
        figsize=(
            18,
            4.2
            *
            num_rows,
        ),
        squeeze=False,
    )

    for row, case_name in enumerate(
        case_names
    ):

        result = results[
            case_name
        ]

        terrain64 = result[
            "data64"
        ][
            "terrain"
        ]

        flat64 = result[
            "data64"
        ][
            "flat"
        ]

        terrain128 = result[
            "data128"
        ][
            "terrain"
        ]

        flat128 = result[
            "data128"
        ][
            "flat"
        ]

        index64 = int(
            np.argmin(
                np.abs(
                    terrain64[
                        "time"
                    ]
                    -
                    plot_time
                )
            )
        )

        index128 = int(
            np.argmin(
                np.abs(
                    terrain128[
                        "time"
                    ]
                    -
                    plot_time
                )
            )
        )

        fields = [
            terrain_speed_field(
                terrain64[
                    "q"
                ],
                flat64[
                    "q"
                ],
                index64,
            ),

            terrain_speed_field(
                result[
                    "cfo"
                ][
                    "terrain"
                ],
                result[
                    "cfo"
                ][
                    "flat"
                ],
                index128,
            ),

            terrain_speed_field(
                result[
                    "final"
                ][
                    "terrain"
                ],
                result[
                    "final"
                ][
                    "flat"
                ],
                index128,
            ),

            terrain_speed_field(
                terrain128[
                    "q"
                ],
                flat128[
                    "q"
                ],
                index128,
            ),
        ]

        labels = [
            METHOD_LABELS[
                "pyclaw64"
            ],

            METHOD_LABELS[
                "cfo"
            ],

            METHOD_LABELS[
                "final"
            ],

            METHOD_LABELS[
                "pyclaw128"
            ],
        ]

        limit = max(
            max(
                float(
                    np.max(
                        np.abs(
                            field
                        )
                    )
                )
                for field
                in fields
            ),
            1.0e-10,
        )

        for column in range(
            4
        ):

            image = heatmap(
                axes[
                    row,
                    column
                ],

                fields[
                    column
                ],

                title=(
                    labels[
                        column
                    ]
                    if row == 0
                    else ""
                ),

                vlim=limit,
            )

            if column == 0:

                axes[
                    row,
                    column
                ].set_ylabel(
                    (
                        case_name
                        +
                        "\ny"
                    )
                )

            fig.colorbar(
                image,

                ax=axes[
                    row,
                    column
                ],

                shrink=0.75,
            )

    fig.suptitle(
        title,
        fontsize=17,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 05
# CROSS-SECTIONS
# =====================================================================

def plot_cross_sections(
    result,
    *,
    plot_time,
    output_path,
):

    terrain64 = result[
        "data64"
    ][
        "terrain"
    ]

    terrain128 = result[
        "data128"
    ][
        "terrain"
    ]

    index64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    index128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    ix64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "x"
                ]
            )
        )
    )

    iy64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "y"
                ]
            )
        )
    )

    ix128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "x"
                ]
            )
        )
    )

    iy128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "y"
                ]
            )
        )
    )

    channels = [
        "h",
        "hu",
        "hv",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            17,
            9,
        ),
    )

    for channel in range(
        3
    ):

        # ---------------------------------------------------------
        # X SECTION
        # ---------------------------------------------------------

        ax = axes[
            0,
            channel
        ]

        ax.plot(
            terrain64[
                "x"
            ],

            terrain64[
                "q"
            ][
                index64,
                :,
                iy64,
                channel
            ],

            label=METHOD_LABELS[
                "pyclaw64"
            ],
        )

        ax.plot(
            terrain128[
                "x"
            ],

            result[
                "cfo"
            ][
                "terrain"
            ][
                index128,
                :,
                iy128,
                channel
            ],

            label=METHOD_LABELS[
                "cfo"
            ],
        )

        ax.plot(
            terrain128[
                "x"
            ],

            result[
                "final"
            ][
                "terrain"
            ][
                index128,
                :,
                iy128,
                channel
            ],

            label=METHOD_LABELS[
                "final"
            ],
        )

        ax.plot(
            terrain128[
                "x"
            ],

            terrain128[
                "q"
            ][
                index128,
                :,
                iy128,
                channel
            ],

            label=METHOD_LABELS[
                "pyclaw128"
            ],
        )

        ax.set_title(
            f"{channels[channel]} — x section"
        )

        ax.set_xlabel(
            "x"
        )

        ax.grid(
            alpha=0.25
        )

        # ---------------------------------------------------------
        # Y SECTION
        # ---------------------------------------------------------

        ax = axes[
            1,
            channel
        ]

        ax.plot(
            terrain64[
                "y"
            ],

            terrain64[
                "q"
            ][
                index64,
                ix64,
                :,
                channel
            ],

            label=METHOD_LABELS[
                "pyclaw64"
            ],
        )

        ax.plot(
            terrain128[
                "y"
            ],

            result[
                "cfo"
            ][
                "terrain"
            ][
                index128,
                ix128,
                :,
                channel
            ],

            label=METHOD_LABELS[
                "cfo"
            ],
        )

        ax.plot(
            terrain128[
                "y"
            ],

            result[
                "final"
            ][
                "terrain"
            ][
                index128,
                ix128,
                :,
                channel
            ],

            label=METHOD_LABELS[
                "final"
            ],
        )

        ax.plot(
            terrain128[
                "y"
            ],

            terrain128[
                "q"
            ][
                index128,
                ix128,
                :,
                channel
            ],

            label=METHOD_LABELS[
                "pyclaw128"
            ],
        )

        ax.set_title(
            f"{channels[channel]} — y section"
        )

        ax.set_xlabel(
            "y"
        )

        ax.grid(
            alpha=0.25
        )

    axes[
        0,
        0
    ].legend(
        fontsize=8
    )

    axes[
        1,
        0
    ].legend(
        fontsize=8
    )

    fig.suptitle(
        (
            "Gaussian-hill cross sections "
            f"at t={terrain128['time'][index128]:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 06
# BATHYMETRY RESOLUTION
# =====================================================================

def plot_bathymetry(
    result,
    *,
    output_path,
):

    terrain64 = result[
        "data64"
    ][
        "terrain"
    ]

    terrain128 = result[
        "data128"
    ][
        "terrain"
    ]

    x64 = terrain64[
        "x"
    ]

    y64 = terrain64[
        "y"
    ]

    x128 = terrain128[
        "x"
    ]

    y128 = terrain128[
        "y"
    ]

    X64, Y64 = np.meshgrid(
        x64,
        y64,
        indexing="ij",
    )

    X128, Y128 = np.meshgrid(
        x128,
        y128,
        indexing="ij",
    )

    iy64 = int(
        np.argmin(
            np.abs(
                y64
            )
        )
    )

    iy128 = int(
        np.argmin(
            np.abs(
                y128
            )
        )
    )

    fig = plt.figure(
        figsize=(
            18,
            5,
        )
    )

    ax1 = fig.add_subplot(
        131,
        projection="3d",
    )

    ax1.plot_surface(
        X64,
        Y64,
        terrain64[
            "b"
        ],
    )

    ax1.set_title(
        "PyClaw 64×64 bathymetry"
    )

    ax1.set_xlabel(
        "x"
    )

    ax1.set_ylabel(
        "y"
    )

    ax1.set_zlabel(
        "b"
    )

    ax2 = fig.add_subplot(
        132,
        projection="3d",
    )

    ax2.plot_surface(
        X128,
        Y128,
        terrain128[
            "b"
        ],
    )

    ax2.set_title(
        (
            "128×128 bathymetry\n"
            "used by zero-shot models + PyClaw128"
        )
    )

    ax2.set_xlabel(
        "x"
    )

    ax2.set_ylabel(
        "y"
    )

    ax2.set_zlabel(
        "b"
    )

    ax3 = fig.add_subplot(
        133
    )

    ax3.plot(
        x64,

        terrain64[
            "b"
        ][
            :,
            iy64
        ],

        label="64×64",
    )

    ax3.plot(
        x128,

        terrain128[
            "b"
        ][
            :,
            iy128
        ],

        label="128×128",
    )

    ax3.set_xlabel(
        "x"
    )

    ax3.set_ylabel(
        "b"
    )

    ax3.set_title(
        "Central bathymetry cross-section"
    )

    ax3.grid(
        alpha=0.25
    )

    ax3.legend()

    fig.suptitle(
        "Bathymetry resolution comparison",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 07
# 3D FREE SURFACE
# =====================================================================

def plot_free_surface(
    result,
    *,
    plot_time,
    output_path,
):

    terrain64 = result[
        "data64"
    ][
        "terrain"
    ]

    terrain128 = result[
        "data128"
    ][
        "terrain"
    ]

    index64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    index128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    X64, Y64 = np.meshgrid(
        terrain64[
            "x"
        ],
        terrain64[
            "y"
        ],
        indexing="ij",
    )

    X128, Y128 = np.meshgrid(
        terrain128[
            "x"
        ],
        terrain128[
            "y"
        ],
        indexing="ij",
    )

    eta64 = (
        terrain64[
            "q"
        ][
            index64,
            ...,
            0
        ]
        +
        terrain64[
            "b"
        ]
    )

    eta_cfo = (
        result[
            "cfo"
        ][
            "terrain"
        ][
            index128,
            ...,
            0
        ]
        +
        terrain128[
            "b"
        ]
    )

    eta_final = (
        result[
            "final"
        ][
            "terrain"
        ][
            index128,
            ...,
            0
        ]
        +
        terrain128[
            "b"
        ]
    )

    eta128 = (
        terrain128[
            "q"
        ][
            index128,
            ...,
            0
        ]
        +
        terrain128[
            "b"
        ]
    )

    entries = [
        (
            METHOD_LABELS[
                "pyclaw64"
            ],
            X64,
            Y64,
            eta64,
        ),

        (
            METHOD_LABELS[
                "cfo"
            ],
            X128,
            Y128,
            eta_cfo,
        ),

        (
            METHOD_LABELS[
                "final"
            ],
            X128,
            Y128,
            eta_final,
        ),

        (
            METHOD_LABELS[
                "pyclaw128"
            ],
            X128,
            Y128,
            eta128,
        ),
    ]

    zmin = min(
        float(
            np.min(
                field
            )
        )
        for _, _, _, field
        in entries
    )

    zmax = max(
        float(
            np.max(
                field
            )
        )
        for _, _, _, field
        in entries
    )

    fig = plt.figure(
        figsize=(
            22,
            5,
        )
    )

    for index, (
        label,
        X,
        Y,
        field,
    ) in enumerate(
        entries,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            4,
            index,
            projection="3d",
        )

        ax.plot_surface(
            X,
            Y,
            field,
        )

        ax.set_zlim(
            zmin,
            zmax,
        )

        ax.set_title(
            label
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

        ax.set_zlabel(
            "η"
        )

    fig.suptitle(
        (
            "Free-surface comparison "
            f"at t={terrain128['time'][index128]:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 08
# 3D STATE CROSS-SECTIONS
# =====================================================================

def plot_3d_state_sections(
    result,
    *,
    plot_time,
    output_path,
):

    terrain64 = result[
        "data64"
    ][
        "terrain"
    ]

    terrain128 = result[
        "data128"
    ][
        "terrain"
    ]

    index64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    index128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "time"
                ]
                -
                plot_time
            )
        )
    )

    iy64 = int(
        np.argmin(
            np.abs(
                terrain64[
                    "y"
                ]
            )
        )
    )

    iy128 = int(
        np.argmin(
            np.abs(
                terrain128[
                    "y"
                ]
            )
        )
    )

    channels = [
        "h",
        "hu",
        "hv",
    ]

    fig = plt.figure(
        figsize=(
            19,
            6,
        )
    )

    entries = [
        (
            METHOD_LABELS[
                "pyclaw64"
            ],

            terrain64[
                "x"
            ],

            terrain64[
                "q"
            ][
                index64,
                :,
                iy64,
            ],

            0.0,
        ),

        (
            METHOD_LABELS[
                "cfo"
            ],

            terrain128[
                "x"
            ],

            result[
                "cfo"
            ][
                "terrain"
            ][
                index128,
                :,
                iy128,
            ],

            1.0,
        ),

        (
            METHOD_LABELS[
                "final"
            ],

            terrain128[
                "x"
            ],

            result[
                "final"
            ][
                "terrain"
            ][
                index128,
                :,
                iy128,
            ],

            2.0,
        ),

        (
            METHOD_LABELS[
                "pyclaw128"
            ],

            terrain128[
                "x"
            ],

            terrain128[
                "q"
            ][
                index128,
                :,
                iy128,
            ],

            3.0,
        ),
    ]

    for channel in range(
        3
    ):

        ax = fig.add_subplot(
            1,
            3,
            channel + 1,
            projection="3d",
        )

        for (
            label,
            x,
            section,
            offset,
        ) in entries:

            ax.plot(
                x,

                np.full_like(
                    x,
                    offset,
                ),

                section[
                    ...,
                    channel
                ],

                label=label,
            )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "Solution"
        )

        ax.set_zlabel(
            channels[
                channel
            ]
        )

        ax.set_yticks(
            [
                0,
                1,
                2,
                3,
            ]
        )

        ax.set_yticklabels(
            [
                "PyClaw64",
                "Bathy-CFO",
                "Canonical",
                "PyClaw128",
            ]
        )

        ax.set_title(
            (
                "3D cross-section: "
                +
                channels[
                    channel
                ]
            )
        )

        ax.legend(
            fontsize=7
        )

    fig.suptitle(
        (
            "Gaussian-hill state cross-sections "
            f"at t={terrain128['time'][index128]:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# METRIC ROW
# =====================================================================

def metric_row(
    case_name,
    method_name,
    truth,
    prediction,
    truth_flat,
    prediction_flat,
):

    return {
        "case":
            case_name,

        "method":
            method_name,

        "rel_l2":
            relative_l2(
                truth,
                prediction,
            ),

        "rmse":
            rmse(
                truth,
                prediction,
            ),

        "h_rel_l2":
            relative_l2(
                truth[
                    ...,
                    0
                ],
                prediction[
                    ...,
                    0
                ],
            ),

        "hu_rel_l2":
            relative_l2(
                truth[
                    ...,
                    1
                ],
                prediction[
                    ...,
                    1
                ],
            ),

        "hv_rel_l2":
            relative_l2(
                truth[
                    ...,
                    2
                ],
                prediction[
                    ...,
                    2
                ],
            ),

        "terrain_effect_rel_l2":
            relative_l2(
                (
                    truth
                    -
                    truth_flat
                ),

                (
                    prediction
                    -
                    prediction_flat
                ),
            ),
    }


# =====================================================================
# SAVE METRICS
# =====================================================================

def save_metrics(
    all_results,
    output_path,
):

    rows = []

    for case_name, result in (
        all_results.items()
    ):

        terrain64 = result[
            "data64"
        ][
            "terrain"
        ]

        flat64 = result[
            "data64"
        ][
            "flat"
        ]

        terrain128 = result[
            "data128"
        ][
            "terrain"
        ]

        flat128 = result[
            "data128"
        ][
            "flat"
        ]

        q64_interp = (
            interpolate_state(
                terrain64[
                    "q"
                ],

                terrain64[
                    "x"
                ],

                terrain64[
                    "y"
                ],

                terrain128[
                    "x"
                ],

                terrain128[
                    "y"
                ],
            )
        )

        flat64_interp = (
            interpolate_state(
                flat64[
                    "q"
                ],

                flat64[
                    "x"
                ],

                flat64[
                    "y"
                ],

                terrain128[
                    "x"
                ],

                terrain128[
                    "y"
                ],
            )
        )

        rows.append(
            metric_row(
                case_name,

                "PyClaw64 interpolated",

                terrain128[
                    "q"
                ],

                q64_interp,

                flat128[
                    "q"
                ],

                flat64_interp,
            )
        )

        rows.append(
            metric_row(
                case_name,

                METHOD_LABELS[
                    "cfo"
                ],

                terrain128[
                    "q"
                ],

                result[
                    "cfo"
                ][
                    "terrain"
                ],

                flat128[
                    "q"
                ],

                result[
                    "cfo"
                ][
                    "flat"
                ],
            )
        )

        rows.append(
            metric_row(
                case_name,

                METHOD_LABELS[
                    "final"
                ],

                terrain128[
                    "q"
                ],

                result[
                    "final"
                ][
                    "terrain"
                ],

                flat128[
                    "q"
                ],

                result[
                    "final"
                ][
                    "flat"
                ],
            )
        )

        rows.append(
            metric_row(
                case_name,

                METHOD_LABELS[
                    "pyclaw128"
                ],

                terrain128[
                    "q"
                ],

                terrain128[
                    "q"
                ],

                flat128[
                    "q"
                ],

                flat128[
                    "q"
                ],
            )
        )

    columns = [
        "case",
        "method",
        "rel_l2",
        "rmse",
        "h_rel_l2",
        "hu_rel_l2",
        "hv_rel_l2",
        "terrain_effect_rel_l2",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # RESTORE MODELS
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "BUILDING FINAL ZERO-SHOT MODELS"
    )

    print(
        "=" * 80
    )

    cfo_method = (
        build_cfo_method()
    )

    cfo_state = (
        restore_cfo_state(
            cfo_method,
            args,
        )
    )

    final_method = (
        build_final_method(
            args
        )
    )

    final_state = (
        restore_final_state(
            final_method,
            args,
        )
    )

    # =================================================================
    # GAUSSIAN CASES
    # =================================================================

    gaussian_results = {}

    for case_name in GAUSSIAN_CASES:

        print()
        print(
            "=" * 80
        )

        print(
            "GAUSSIAN:",
            case_name
        )

        print(
            "=" * 80
        )

        data64 = load_pair(
            args.gaussian64,
            case_name,
        )

        data128 = load_pair(
            args.gaussian128,
            case_name,
        )

        gaussian_results[
            case_name
        ] = evaluate_case(
            data64,
            data128,

            cfo_method=(
                cfo_method
            ),

            cfo_state=(
                cfo_state
            ),

            final_method=(
                final_method
            ),

            final_state=(
                final_state
            ),

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

    # =================================================================
    # OOD CASES
    # =================================================================

    ood_results = {}

    for case_name in OOD_CASES:

        print()
        print(
            "=" * 80
        )

        print(
            "OOD:",
            case_name
        )

        print(
            "=" * 80
        )

        data64 = load_pair(
            args.ood64,
            case_name,
        )

        data128 = load_pair(
            args.ood128,
            case_name,
        )

        ood_results[
            case_name
        ] = evaluate_case(
            data64,
            data128,

            cfo_method=(
                cfo_method
            ),

            cfo_state=(
                cfo_state
            ),

            final_method=(
                final_method
            ),

            final_state=(
                final_state
            ),

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

    # =================================================================
    # FIGURE 01
    # =================================================================

    print(
        "\nCreating Figure 01..."
    )

    plot_direct_condition(
        gaussian_results[
            args.direct_case
        ],

        cfo_method=(
            cfo_method
        ),

        cfo_state=(
            cfo_state
        ),

        final_method=(
            final_method
        ),

        final_state=(
            final_state
        ),

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "01_direct_condition_test.png"
        ),
    )

    # =================================================================
    # FIGURE 02
    # =================================================================

    print(
        "Creating Figure 02..."
    )

    plot_isolated_hill_effect(
        gaussian_results,

        output_path=(
            output_dir
            /
            "02_isolated_hill_effect_error.png"
        ),
    )

    # =================================================================
    # FIGURE 03
    # =================================================================

    print(
        "Creating Figure 03..."
    )

    plot_speed_grid(
        gaussian_results,

        GAUSSIAN_CASES,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "03_terrain_induced_speed_gaussian.png"
        ),

        title=(
            "Zero-shot terrain-induced speed "
            "for Gaussian hills"
        ),
    )

    # =================================================================
    # FIGURE 04
    # =================================================================

    print(
        "Creating Figure 04..."
    )

    plot_speed_grid(
        ood_results,

        OOD_CASES,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "04_terrain_induced_speed_unseen.png"
        ),

        title=(
            "Zero-shot terrain-induced speed "
            "on unseen bathymetry"
        ),
    )

    cross_result = (
        gaussian_results[
            args.cross_section_case
        ]
    )

    # =================================================================
    # FIGURE 05
    # =================================================================

    print(
        "Creating Figure 05..."
    )

    plot_cross_sections(
        cross_result,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "05_gaussian_cross_sections_h_hu_hv.png"
        ),
    )

    # =================================================================
    # FIGURE 06
    # =================================================================

    print(
        "Creating Figure 06..."
    )

    plot_bathymetry(
        cross_result,

        output_path=(
            output_dir
            /
            "06_gaussian_3d_bathymetry_cross_sections.png"
        ),
    )

    # =================================================================
    # FIGURE 07
    # =================================================================

    print(
        "Creating Figure 07..."
    )

    plot_free_surface(
        cross_result,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "07_gaussian_3d_free_surface.png"
        ),
    )

    # =================================================================
    # FIGURE 08
    # =================================================================

    print(
        "Creating Figure 08..."
    )

    plot_3d_state_sections(
        cross_result,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "08_gaussian_3d_h_hu_hv_cross_sections.png"
        ),
    )

    # =================================================================
    # METRICS
    # =================================================================

    all_results = {}

    for case_name, result in (
        gaussian_results.items()
    ):

        all_results[
            "gaussian_"
            +
            case_name
        ] = result

    for case_name, result in (
        ood_results.items()
    ):

        all_results[
            "ood_"
            +
            case_name
        ] = result

    save_metrics(
        all_results,

        output_dir
        /
        "zero_shot_metrics.csv",
    )

    # =================================================================
    # COMPLETE
    # =================================================================

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL ZERO-SHOT STUDY COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        "Output:"
    )

    print(
        output_dir
    )


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":

    main()