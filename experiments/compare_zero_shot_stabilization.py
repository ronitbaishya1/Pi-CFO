"""
Zero-shot resolution-transfer comparison.

Compare:

1. PyClaw 64x64 interpolated to 128x128
2. Original Geometry-U-FNO trained at 64, evaluated at 128
3. Canonical-U-Net Geometry-U-FNO
4. Canonical local Geometry-U-FNO
       - U-Net evaluated at canonical 64
       - geometry encoder evaluated at canonical 64

Reference:
    PyClaw 128x128

IMPORTANT
---------
No 128x128 model training is performed.
The same 64x64 trained checkpoint is used for every neural model.
"""

from __future__ import annotations

import argparse
import csv

from pathlib import Path

import h5py

import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np


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
# PROJECT
# =====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# =====================================================================
# DOMAIN
# =====================================================================

XMIN = -2.5
XMAX = 2.5

YMIN = -2.5
YMAX = 2.5

DOMAIN_LENGTH_X = (
    XMAX
    -
    XMIN
)

DOMAIN_LENGTH_Y = (
    YMAX
    -
    YMIN
)


# =====================================================================
# RESOLUTION
# =====================================================================

SOURCE_RESOLUTION = 64

TARGET_RESOLUTION = 128


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
        "--checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "rollout_stable/"
            "geometry_ufno/"
            "res64/"
            "K2_lamroll_0p1_lamhf_0/"
            "seed0/"
            "best"
        ),
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
        "--output-dir",
        type=str,
        default=(
            "results/"
            "zero_shot_stabilization/"
            "canonical_local_branches"
        ),
    )

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve(
    value,
):

    path = (
        Path(
            value
        )
        .expanduser()
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# DATA LOADING
# =====================================================================

def load_case(
    file_path,
    case_name,
):

    file_path = resolve(
        file_path
    )

    with h5py.File(
        file_path,
        "r",
    ) as h5:

        q = np.asarray(
            h5[
                f"cases/{case_name}/q"
            ],
            dtype=np.float32,
        )

        bathymetry = np.asarray(
            h5[
                f"cases/{case_name}/bathymetry"
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

    if (
        bathymetry.ndim
        ==
        3
        and
        bathymetry.shape[
            -1
        ]
        ==
        1
    ):

        bathymetry = (
            bathymetry[
                ...,
                0
            ]
        )

    return {
        "q":
            q,

        "b":
            bathymetry,

        "time":
            time,

        "x":
            x,

        "y":
            y,
    }


# =====================================================================
# 2D INTERPOLATION
# =====================================================================

def interp2(
    field,
    x_source,
    y_source,
    x_target,
    y_target,
):

    intermediate = np.empty(
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

    # -----------------------------------------------------------------
    # X INTERPOLATION
    # -----------------------------------------------------------------

    for j in range(
        len(
            y_source
        )
    ):

        intermediate[
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

    # -----------------------------------------------------------------
    # Y INTERPOLATION
    # -----------------------------------------------------------------

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
            intermediate[
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

    num_times = (
        q.shape[
            0
        ]
    )

    num_channels = (
        q.shape[
            -1
        ]
    )

    output = np.empty(
        (
            num_times,
            len(
                x_target
            ),
            len(
                y_target
            ),
            num_channels,
        ),
        dtype=np.float32,
    )

    for time_index in range(
        num_times
    ):

        for channel in range(
            num_channels
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
# MODEL BUILDER
# =====================================================================

def build_method(
    *,
    canonical_unet_resolution,
    canonical_geometry_resolution,
):

    resolution = (
        TARGET_RESOLUTION
    )

    dx = (
        DOMAIN_LENGTH_X
        /
        resolution
    )

    dy = (
        DOMAIN_LENGTH_Y
        /
        resolution
    )

    model = (
        GeometryUFNO2d(
            num_channels=3,

            modes1=12,

            modes2=12,

            width=64,

            num_blocks=4,

            num_u_blocks=2,

            geometry_width=16,

            geometry_depth=2,

            dx=dx,

            dy=dy,

            include_gradient_magnitude=False,

            use_time=True,

            canonical_unet_resolution=(
                canonical_unet_resolution
            ),

            canonical_geometry_resolution=(
                canonical_geometry_resolution
            ),

            # ---------------------------------------------------------
            # LOW-PASS FILTER OFF
            # ---------------------------------------------------------

            source_resolution_filter=None,

            domain_length_x=(
                DOMAIN_LENGTH_X
            ),

            domain_length_y=(
                DOMAIN_LENGTH_Y
            ),
        )
    )

    method = (
        WellBalancedBathymetryBedPICFO(
            model=model,

            input_shape=(
                resolution,
                resolution,
                3,
            ),

            condition_shape=(
                resolution,
                resolution,
                1,
            ),

            gamma=1.0e-5,

            spline_type="quintic",

            lambda_pde=0.01,

            lambda_bed=0.70,

            lambda_wb=0.10,

            wb_eta0=1.5,

            dx=dx,

            dy=dy,

            gravity=1.0,
        )
    )

    return method


# =====================================================================
# CHECKPOINT
# =====================================================================

def restore_state(
    method,
    checkpoint,
):

    state = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,

        beta2=0.99,
    )

    checkpoint = resolve(
        checkpoint
    )

    state = load_train_state(
        state,

        ckpt_dir=str(
            checkpoint.parent
        ),

        prefix=(
            checkpoint.name
        ),

        step=None,

        max_to_keep=1,
    )

    return state


# =====================================================================
# ROLLOUT SHAPE
# =====================================================================

def normalize_prediction(
    prediction,
    expected_times,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    # -----------------------------------------------------------------
    # (T,1,H,W,C)
    # -----------------------------------------------------------------

    if (
        prediction.ndim
        ==
        5
        and
        prediction.shape[
            0
        ]
        ==
        expected_times
        and
        prediction.shape[
            1
        ]
        ==
        1
    ):

        return prediction[
            :,
            0
        ]

    # -----------------------------------------------------------------
    # (1,T,H,W,C)
    # -----------------------------------------------------------------

    if (
        prediction.ndim
        ==
        5
        and
        prediction.shape[
            0
        ]
        ==
        1
        and
        prediction.shape[
            1
        ]
        ==
        expected_times
    ):

        return prediction[
            0
        ]

    # -----------------------------------------------------------------
    # (T,H,W,C)
    # -----------------------------------------------------------------

    if (
        prediction.ndim
        ==
        4
    ):

        return prediction

    raise ValueError(
        "Unexpected rollout shape: "
        f"{prediction.shape}"
    )


# =====================================================================
# ROLLOUT
# =====================================================================

def rollout(
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

    prediction = (
        method.uniform_inference(
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

            condition=(
                condition
            ),

            method="RK4",
        )
    )

    return normalize_prediction(
        prediction,
        q_reference.shape[
            0
        ],
    )


# =====================================================================
# METRICS
# =====================================================================

def relative_l2(
    truth,
    prediction,
):

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
            1.0e-12,
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
                    prediction
                    -
                    truth
                )
                ** 2
            )
        )
    )


def state_metrics(
    truth,
    prediction,
):

    return {

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

        "final_rel_l2":
            relative_l2(
                truth[
                    -1
                ],
                prediction[
                    -1
                ],
            ),
    }


def terrain_effect_error(
    terrain_truth,
    flat_truth,
    terrain_prediction,
    flat_prediction,
):

    true_effect = (
        terrain_truth
        -
        flat_truth
    )

    predicted_effect = (
        terrain_prediction
        -
        flat_prediction
    )

    return relative_l2(
        true_effect,
        predicted_effect,
    )


def relative_error_vs_time(
    truth,
    prediction,
):

    values = []

    for time_index in range(
        truth.shape[
            0
        ]
    ):

        numerator = np.linalg.norm(
            (
                prediction[
                    time_index
                ]
                -
                truth[
                    time_index
                ]
            ).ravel()
        )

        denominator = np.linalg.norm(
            truth[
                time_index
            ].ravel()
        )

        if denominator < 1.0e-12:

            if numerator < 1.0e-12:

                values.append(
                    0.0
                )

            else:

                values.append(
                    numerator
                )

        else:

            values.append(
                numerator
                /
                denominator
            )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# =====================================================================
# SPEED
# =====================================================================

def speed_from_state(
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
# MAIN
# =====================================================================

def main():

    args = parse_args()

    output_dir = resolve(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # LOAD GAUSSIAN
    # =================================================================

    gaussian64 = load_case(
        args.gaussian64,
        "hill_center",
    )

    flat64 = load_case(
        args.gaussian64,
        "flat",
    )

    gaussian128 = load_case(
        args.gaussian128,
        "hill_center",
    )

    flat128 = load_case(
        args.gaussian128,
        "flat",
    )

    # =================================================================
    # LOAD OOD
    # =================================================================

    ood64 = load_case(
        args.ood64,
        "two_hills_00",
    )

    ood_flat64 = load_case(
        args.ood64,
        "flat",
    )

    ood128 = load_case(
        args.ood128,
        "two_hills_00",
    )

    ood_flat128 = load_case(
        args.ood128,
        "flat",
    )

    # =================================================================
    # PYCLAW 64 -> 128 INTERPOLATION
    # =================================================================

    gaussian64_interp = (
        interpolate_state(
            gaussian64[
                "q"
            ],

            gaussian64[
                "x"
            ],

            gaussian64[
                "y"
            ],

            gaussian128[
                "x"
            ],

            gaussian128[
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

            flat128[
                "x"
            ],

            flat128[
                "y"
            ],
        )
    )

    ood64_interp = (
        interpolate_state(
            ood64[
                "q"
            ],

            ood64[
                "x"
            ],

            ood64[
                "y"
            ],

            ood128[
                "x"
            ],

            ood128[
                "y"
            ],
        )
    )

    ood_flat64_interp = (
        interpolate_state(
            ood_flat64[
                "q"
            ],

            ood_flat64[
                "x"
            ],

            ood_flat64[
                "y"
            ],

            ood_flat128[
                "x"
            ],

            ood_flat128[
                "y"
            ],
        )
    )

    # =================================================================
    # MODEL CONFIGURATIONS
    # =================================================================

    configurations = {

        "baseline": {

            "label":
                "Original Geo-U-FNO 64→128",

            "canonical_unet":
                None,

            "canonical_geometry":
                None,
        },

        "canonical_unet": {

            "label":
                "Canonical-U-Net 64→128",

            "canonical_unet":
                64,

            "canonical_geometry":
                None,
        },

        "canonical_local": {

            "label":
                "Canonical Local Branches 64→128",

            "canonical_unet":
                64,

            "canonical_geometry":
                64,
        },
    }

    # =================================================================
    # RUN MODEL VARIANTS
    # =================================================================

    predictions = {}

    for (
        key,
        configuration,
    ) in configurations.items():

        print()

        print(
            "=" * 80
        )

        print(
            configuration[
                "label"
            ]
        )

        print(
            "=" * 80
        )

        print(
            "Canonical U-Net:",
            configuration[
                "canonical_unet"
            ],
        )

        print(
            "Canonical geometry:",
            configuration[
                "canonical_geometry"
            ],
        )

        # -------------------------------------------------------------
        # METHOD
        # -------------------------------------------------------------

        method = build_method(
            canonical_unet_resolution=(
                configuration[
                    "canonical_unet"
                ]
            ),

            canonical_geometry_resolution=(
                configuration[
                    "canonical_geometry"
                ]
            ),
        )

        # -------------------------------------------------------------
        # SAME 64 CHECKPOINT
        # -------------------------------------------------------------

        state = restore_state(
            method,
            args.checkpoint,
        )

        # -------------------------------------------------------------
        # GAUSSIAN
        # -------------------------------------------------------------

        gaussian_prediction = rollout(
            method,
            state,

            gaussian128[
                "q"
            ],

            gaussian128[
                "b"
            ],

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        flat_prediction = rollout(
            method,
            state,

            flat128[
                "q"
            ],

            flat128[
                "b"
            ],

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        # -------------------------------------------------------------
        # OOD
        # -------------------------------------------------------------

        ood_prediction = rollout(
            method,
            state,

            ood128[
                "q"
            ],

            ood128[
                "b"
            ],

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        ood_flat_prediction = rollout(
            method,
            state,

            ood_flat128[
                "q"
            ],

            ood_flat128[
                "b"
            ],

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        predictions[
            key
        ] = {

            "gaussian":
                gaussian_prediction,

            "flat":
                flat_prediction,

            "ood":
                ood_prediction,

            "ood_flat":
                ood_flat_prediction,
        }

    # =================================================================
    # NUMERICAL TABLE
    # =================================================================

    rows = []

    # -----------------------------------------------------------------
    # PYCLAW RESOLUTION DIFFERENCE
    # -----------------------------------------------------------------

    pyclaw_metrics = state_metrics(
        gaussian128[
            "q"
        ],
        gaussian64_interp,
    )

    pyclaw_metrics[
        "gaussian_effect_rel_l2"
    ] = terrain_effect_error(
        gaussian128[
            "q"
        ],

        flat128[
            "q"
        ],

        gaussian64_interp,

        flat64_interp,
    )

    pyclaw_metrics[
        "ood_effect_rel_l2"
    ] = terrain_effect_error(
        ood128[
            "q"
        ],

        ood_flat128[
            "q"
        ],

        ood64_interp,

        ood_flat64_interp,
    )

    rows.append(
        {
            "model":
                "PyClaw64 interpolated",

            **pyclaw_metrics,
        }
    )

    # -----------------------------------------------------------------
    # NEURAL MODELS
    # -----------------------------------------------------------------

    for (
        key,
        configuration,
    ) in configurations.items():

        current = predictions[
            key
        ]

        current_metrics = state_metrics(
            gaussian128[
                "q"
            ],

            current[
                "gaussian"
            ],
        )

        current_metrics[
            "gaussian_effect_rel_l2"
        ] = terrain_effect_error(
            gaussian128[
                "q"
            ],

            flat128[
                "q"
            ],

            current[
                "gaussian"
            ],

            current[
                "flat"
            ],
        )

        current_metrics[
            "ood_effect_rel_l2"
        ] = terrain_effect_error(
            ood128[
                "q"
            ],

            ood_flat128[
                "q"
            ],

            current[
                "ood"
            ],

            current[
                "ood_flat"
            ],
        )

        rows.append(
            {
                "model":
                    configuration[
                        "label"
                    ],

                **current_metrics,
            }
        )

    # -----------------------------------------------------------------
    # SAVE TABLE
    # -----------------------------------------------------------------

    columns = [

        "model",

        "rel_l2",

        "rmse",

        "h_rel_l2",

        "hu_rel_l2",

        "hv_rel_l2",

        "final_rel_l2",

        "gaussian_effect_rel_l2",

        "ood_effect_rel_l2",
    ]

    csv_path = (
        output_dir
        /
        "zero_shot_canonical_local_metrics.csv"
    )

    with open(
        csv_path,
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

    # =================================================================
    # PLOT TIME
    # =================================================================

    plot_index = int(
        np.argmin(
            np.abs(
                gaussian128[
                    "time"
                ]
                -
                args.plot_time
            )
        )
    )

    actual_plot_time = float(
        gaussian128[
            "time"
        ][
            plot_index
        ]
    )

    # =================================================================
    # FIGURE 1
    # GAUSSIAN TERRAIN-EFFECT ERROR VS TIME
    # =================================================================

    gaussian_true_effect = (
        gaussian128[
            "q"
        ]
        -
        flat128[
            "q"
        ]
    )

    gaussian_pyclaw64_effect = (
        gaussian64_interp
        -
        flat64_interp
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.plot(
        gaussian128[
            "time"
        ],

        relative_error_vs_time(
            gaussian_true_effect,
            gaussian_pyclaw64_effect,
        ),

        label=(
            "PyClaw64 interp → PyClaw128"
        ),
    )

    for (
        key,
        configuration,
    ) in configurations.items():

        predicted_effect = (
            predictions[
                key
            ][
                "gaussian"
            ]
            -
            predictions[
                key
            ][
                "flat"
            ]
        )

        ax.plot(
            gaussian128[
                "time"
            ],

            relative_error_vs_time(
                gaussian_true_effect,
                predicted_effect,
            ),

            label=(
                configuration[
                    "label"
                ]
            ),
        )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Relative terrain-effect error"
    )

    ax.set_title(
        "Gaussian terrain-effect error"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "01_gaussian_effect_error_vs_time.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 2
    # OOD TERRAIN-EFFECT ERROR VS TIME
    # =================================================================

    ood_true_effect = (
        ood128[
            "q"
        ]
        -
        ood_flat128[
            "q"
        ]
    )

    ood_pyclaw64_effect = (
        ood64_interp
        -
        ood_flat64_interp
    )

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.plot(
        ood128[
            "time"
        ],

        relative_error_vs_time(
            ood_true_effect,
            ood_pyclaw64_effect,
        ),

        label=(
            "PyClaw64 interp → PyClaw128"
        ),
    )

    for (
        key,
        configuration,
    ) in configurations.items():

        predicted_effect = (
            predictions[
                key
            ][
                "ood"
            ]
            -
            predictions[
                key
            ][
                "ood_flat"
            ]
        )

        ax.plot(
            ood128[
                "time"
            ],

            relative_error_vs_time(
                ood_true_effect,
                predicted_effect,
            ),

            label=(
                configuration[
                    "label"
                ]
            ),
        )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Relative terrain-effect error"
    )

    ax.set_title(
        "OOD two-hill terrain-effect error"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "02_ood_effect_error_vs_time.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 3
    # CROSS SECTIONS
    # =================================================================

    x_index = int(
        np.argmin(
            np.abs(
                gaussian128[
                    "x"
                ]
            )
        )
    )

    y_index = int(
        np.argmin(
            np.abs(
                gaussian128[
                    "y"
                ]
            )
        )
    )

    channel_names = [
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

        # -------------------------------------------------------------
        # X SECTION
        # -------------------------------------------------------------

        ax = axes[
            0,
            channel
        ]

        ax.plot(
            gaussian128[
                "x"
            ],

            gaussian128[
                "q"
            ][
                plot_index,
                :,
                y_index,
                channel
            ],

            label=(
                "PyClaw128"
            ),
        )

        ax.plot(
            gaussian128[
                "x"
            ],

            gaussian64_interp[
                plot_index,
                :,
                y_index,
                channel
            ],

            label=(
                "PyClaw64 interp"
            ),
        )

        for (
            key,
            configuration,
        ) in configurations.items():

            ax.plot(
                gaussian128[
                    "x"
                ],

                predictions[
                    key
                ][
                    "gaussian"
                ][
                    plot_index,
                    :,
                    y_index,
                    channel
                ],

                label=(
                    configuration[
                        "label"
                    ]
                ),
            )

        ax.set_xlabel(
            "x"
        )

        ax.set_title(
            (
                f"{channel_names[channel]} "
                "— x section"
            )
        )

        ax.grid(
            alpha=0.25
        )

        # -------------------------------------------------------------
        # Y SECTION
        # -------------------------------------------------------------

        ax = axes[
            1,
            channel
        ]

        ax.plot(
            gaussian128[
                "y"
            ],

            gaussian128[
                "q"
            ][
                plot_index,
                x_index,
                :,
                channel
            ],

            label=(
                "PyClaw128"
            ),
        )

        ax.plot(
            gaussian128[
                "y"
            ],

            gaussian64_interp[
                plot_index,
                x_index,
                :,
                channel
            ],

            label=(
                "PyClaw64 interp"
            ),
        )

        for (
            key,
            configuration,
        ) in configurations.items():

            ax.plot(
                gaussian128[
                    "y"
                ],

                predictions[
                    key
                ][
                    "gaussian"
                ][
                    plot_index,
                    x_index,
                    :,
                    channel
                ],

                label=(
                    configuration[
                        "label"
                    ]
                ),
            )

        ax.set_xlabel(
            "y"
        )

        ax.set_title(
            (
                f"{channel_names[channel]} "
                "— y section"
            )
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
            "64→128 zero-shot local-branch comparison "
            f"at t={actual_plot_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "03_gaussian_cross_sections.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 4
    # FREE SURFACE
    # =================================================================

    free_surface_fields = [

        (
            "PyClaw128",

            gaussian128[
                "q"
            ][
                plot_index,
                ...,
                0
            ]
            +
            gaussian128[
                "b"
            ],
        ),
    ]

    for (
        key,
        configuration,
    ) in configurations.items():

        eta = (
            predictions[
                key
            ][
                "gaussian"
            ][
                plot_index,
                ...,
                0
            ]
            +
            gaussian128[
                "b"
            ]
        )

        free_surface_fields.append(
            (
                configuration[
                    "label"
                ],

                eta,
            )
        )

    eta_min = min(
        float(
            np.min(
                field
            )
        )

        for _, field
        in free_surface_fields
    )

    eta_max = max(
        float(
            np.max(
                field
            )
        )

        for _, field
        in free_surface_fields
    )

    fig, axes = plt.subplots(
        1,
        len(
            free_surface_fields
        ),
        figsize=(
            19,
            4.8,
        ),
    )

    for (
        ax,
        (
            label,
            field,
        ),
    ) in zip(
        axes,
        free_surface_fields,
    ):

        image = ax.imshow(
            field.T,

            origin="lower",

            extent=[
                XMIN,
                XMAX,
                YMIN,
                YMAX,
            ],

            vmin=eta_min,

            vmax=eta_max,

            aspect="equal",
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

        fig.colorbar(
            image,
            ax=ax,
            shrink=0.8,
        )

    fig.suptitle(
        (
            "Free-surface comparison "
            f"at t={actual_plot_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "04_free_surface_comparison.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 5
    # GAUSSIAN TERRAIN-INDUCED SPEED
    # =================================================================

    truth_speed_effect = (
        speed_from_state(
            gaussian128[
                "q"
            ][
                plot_index
            ]
        )
        -
        speed_from_state(
            flat128[
                "q"
            ][
                plot_index
            ]
        )
    )

    speed_fields = [

        (
            "PyClaw128",

            truth_speed_effect,
        ),
    ]

    for (
        key,
        configuration,
    ) in configurations.items():

        predicted_speed_effect = (
            speed_from_state(
                predictions[
                    key
                ][
                    "gaussian"
                ][
                    plot_index
                ]
            )
            -
            speed_from_state(
                predictions[
                    key
                ][
                    "flat"
                ][
                    plot_index
                ]
            )
        )

        speed_fields.append(
            (
                configuration[
                    "label"
                ],

                predicted_speed_effect,
            )
        )

    speed_limit = max(
        float(
            np.max(
                np.abs(
                    field
                )
            )
        )

        for _, field
        in speed_fields
    )

    fig, axes = plt.subplots(
        1,
        len(
            speed_fields
        ),
        figsize=(
            19,
            4.8,
        ),
    )

    for (
        ax,
        (
            label,
            field,
        ),
    ) in zip(
        axes,
        speed_fields,
    ):

        image = ax.imshow(
            field.T,

            origin="lower",

            extent=[
                XMIN,
                XMAX,
                YMIN,
                YMAX,
            ],

            cmap="coolwarm",

            vmin=(
                -speed_limit
            ),

            vmax=(
                speed_limit
            ),

            aspect="equal",
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

        fig.colorbar(
            image,
            ax=ax,
            shrink=0.8,
        )

    fig.suptitle(
        (
            "Gaussian terrain-induced speed "
            f"at t={actual_plot_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "05_terrain_speed_comparison.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # PRINT TABLE
    # =================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "ZERO-SHOT CANONICAL LOCAL-BRANCH RESULTS"
    )

    print(
        "=" * 90
    )

    for row in rows:

        print()

        print(
            row[
                "model"
            ]
        )

        for column in columns[
            1:
        ]:

            print(
                f"  {column:30s}: "
                f"{row[column]:.8f}"
            )

    print()

    print(
        "CSV:"
    )

    print(
        csv_path
    )

    print()

    print(
        "Results directory:"
    )

    print(
        output_dir
    )


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":

    main()