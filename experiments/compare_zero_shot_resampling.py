"""
Compare zero-shot U-Net resampling strategies.

All neural models use:

    same 64x64 trained checkpoint
    same Geometry-U-FNO weights
    no 128x128 training
    canonical U-Net resolution = 64
    geometry encoder remains native 128
    RK4 steps per segment = configurable

Strategies
----------

baseline:
    original Geometry-U-FNO evaluated directly at 128

area_linear:
    128 -> 64 using 2x2 area averaging
    U-Net at 64
    64 -> 128 using linear interpolation

area_cubic:
    128 -> 64 using 2x2 area averaging
    U-Net at 64
    64 -> 128 using cubic interpolation

area_cubic_blend:
    same as area_cubic
    plus a small contribution from the native 128 U-Net branch

Reference:
    PyClaw 128x128
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

LX = (
    XMAX
    -
    XMIN
)

LY = (
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
        "--native-blend",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "zero_shot_resampling/"
            "64_to_128"
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
# LOAD DATA
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

        b = np.asarray(
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
        b.ndim == 3
        and
        b.shape[-1] == 1
    ):

        b = b[
            ...,
            0
        ]

    return {
        "q": q,
        "b": b,
        "time": time,
        "x": x,
        "y": y,
    }


# =====================================================================
# INTERPOLATION
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

    # -----------------------------------------------------------------
    # X
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Y
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

    nt = (
        q.shape[
            0
        ]
    )

    nc = (
        q.shape[
            -1
        ]
    )

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
# BUILD MODEL
# =====================================================================

def build_method(
    *,
    canonical_resolution,
    downsample_mode,
    upsample_mode,
    native_blend,
):

    resolution = (
        TARGET_RESOLUTION
    )

    dx = (
        LX
        /
        resolution
    )

    dy = (
        LY
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

            # ---------------------------------------------------------
            # CANONICAL U-NET
            # ---------------------------------------------------------

            canonical_unet_resolution=(
                canonical_resolution
            ),

            # ---------------------------------------------------------
            # GEOMETRY ENCODER REMAINS NATIVE 128
            # ---------------------------------------------------------

            canonical_geometry_resolution=None,

            # ---------------------------------------------------------
            # RESAMPLING
            # ---------------------------------------------------------

            canonical_downsample_mode=(
                downsample_mode
            ),

            canonical_upsample_mode=(
                upsample_mode
            ),

            canonical_native_blend=(
                native_blend
            ),

            # ---------------------------------------------------------
            # NO SOURCE LP FILTER
            # ---------------------------------------------------------

            source_resolution_filter=None,

            domain_length_x=LX,

            domain_length_y=LY,
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

    return load_train_state(
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


# =====================================================================
# PREDICTION SHAPE
# =====================================================================

def normalize_prediction(
    prediction,
    nt,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if (
        prediction.ndim == 5
        and
        prediction.shape[0] == nt
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
        prediction.shape[1] == nt
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

            condition=condition,

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


def effect_error(
    terrain_truth,
    flat_truth,
    terrain_prediction,
    flat_prediction,
):

    return relative_l2(
        (
            terrain_truth
            -
            flat_truth
        ),
        (
            terrain_prediction
            -
            flat_prediction
        ),
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

def speed_from_q(
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
    # LOAD DATA
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
    # PYCLAW 64 -> 128 REFERENCE DIFFERENCE
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
    # STRATEGIES
    # =================================================================

    configurations = {

        "baseline": {

            "label":
                "Original Geo-U-FNO",

            "canonical":
                None,

            "down":
                "area",

            "up":
                "linear",

            "blend":
                0.0,
        },

        "area_linear": {

            "label":
                "Canonical: Area↓ + Linear↑",

            "canonical":
                64,

            "down":
                "area",

            "up":
                "linear",

            "blend":
                0.0,
        },

        "area_cubic": {

            "label":
                "Canonical: Area↓ + Cubic↑",

            "canonical":
                64,

            "down":
                "area",

            "up":
                "cubic",

            "blend":
                0.0,
        },

        "area_cubic_blend": {

            "label":
                (
                    "Canonical: Area↓ + Cubic↑ "
                    f"+ {args.native_blend:.2f} native"
                ),

            "canonical":
                64,

            "down":
                "area",

            "up":
                "cubic",

            "blend":
                float(
                    args.native_blend
                ),
        },
    }

    # =================================================================
    # RUN
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

        method = build_method(
            canonical_resolution=(
                configuration[
                    "canonical"
                ]
            ),

            downsample_mode=(
                configuration[
                    "down"
                ]
            ),

            upsample_mode=(
                configuration[
                    "up"
                ]
            ),

            native_blend=(
                configuration[
                    "blend"
                ]
            ),
        )

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
    # TABLE
    # =================================================================

    rows = []

    # -----------------------------------------------------------------
    # PYCLAW 64 -> 128 FLOOR
    # -----------------------------------------------------------------

    floor_metrics = state_metrics(
        gaussian128[
            "q"
        ],
        gaussian64_interp,
    )

    floor_metrics[
        "gaussian_effect_rel_l2"
    ] = effect_error(
        gaussian128[
            "q"
        ],

        flat128[
            "q"
        ],

        gaussian64_interp,

        flat64_interp,
    )

    floor_metrics[
        "ood_effect_rel_l2"
    ] = effect_error(
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

            **floor_metrics,
        }
    )

    # -----------------------------------------------------------------
    # MODELS
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
        ] = effect_error(
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
        ] = effect_error(
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
    # SAVE CSV
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
        "zero_shot_resampling_metrics.csv"
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
    # TIME
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

    actual_time = float(
        gaussian128[
            "time"
        ][
            plot_index
        ]
    )

    # =================================================================
    # FIGURE 01
    # GAUSSIAN EFFECT ERROR
    # =================================================================

    true_gaussian_effect = (
        gaussian128[
            "q"
        ]
        -
        flat128[
            "q"
        ]
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
            true_gaussian_effect,

            (
                gaussian64_interp
                -
                flat64_interp
            ),
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
                true_gaussian_effect,
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

    ax.legend(
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "01_gaussian_effect_error.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 02
    # OOD EFFECT
    # =================================================================

    true_ood_effect = (
        ood128[
            "q"
        ]
        -
        ood_flat128[
            "q"
        ]
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
            true_ood_effect,

            (
                ood64_interp
                -
                ood_flat64_interp
            ),
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
                true_ood_effect,
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

    ax.legend(
        fontsize=9
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "02_ood_effect_error.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 03
    # CROSS-SECTIONS
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
            18,
            9,
        ),
    )

    for channel in range(
        3
    ):

        # -------------------------------------------------------------
        # X
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

            label="PyClaw128",
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

            label="PyClaw64 interp",
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

        ax.set_title(
            (
                f"{channel_names[channel]} "
                "— x section"
            )
        )

        ax.set_xlabel(
            "x"
        )

        ax.grid(
            alpha=0.25
        )

        # -------------------------------------------------------------
        # Y
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

            label="PyClaw128",
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

            label="PyClaw64 interp",
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

        ax.set_title(
            (
                f"{channel_names[channel]} "
                "— y section"
            )
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
        fontsize=7
    )

    axes[
        1,
        0
    ].legend(
        fontsize=7
    )

    fig.suptitle(
        (
            "64→128 canonical-resampling comparison "
            f"at t={actual_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "03_cross_sections.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 04
    # FREE SURFACE
    # =================================================================

    fields = [

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

        fields.append(
            (
                configuration[
                    "label"
                ],

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
                ],
            )
        )

    eta_min = min(
        float(
            np.min(
                field
            )
        )

        for _, field
        in fields
    )

    eta_max = max(
        float(
            np.max(
                field
            )
        )

        for _, field
        in fields
    )

    fig, axes = plt.subplots(
        1,
        len(
            fields
        ),
        figsize=(
            22,
            4.5,
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
        fields,
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
            label,
            fontsize=9,
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
            f"at t={actual_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "04_free_surface.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # FIGURE 05
    # TERRAIN-INDUCED SPEED
    # =================================================================

    speed_fields = [

        (
            "PyClaw128",

            (
                speed_from_q(
                    gaussian128[
                        "q"
                    ][
                        plot_index
                    ]
                )
                -
                speed_from_q(
                    flat128[
                        "q"
                    ][
                        plot_index
                    ]
                )
            ),
        ),
    ]

    for (
        key,
        configuration,
    ) in configurations.items():

        speed_fields.append(
            (
                configuration[
                    "label"
                ],

                (
                    speed_from_q(
                        predictions[
                            key
                        ][
                            "gaussian"
                        ][
                            plot_index
                        ]
                    )
                    -
                    speed_from_q(
                        predictions[
                            key
                        ][
                            "flat"
                        ][
                            plot_index
                        ]
                    )
                ),
            )
        )

    limit = max(
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
            22,
            4.5,
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

            vmin=-limit,

            vmax=limit,

            aspect="equal",
        )

        ax.set_title(
            label,
            fontsize=9,
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
            f"at t={actual_time:.2f}"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "05_terrain_speed.png",

        dpi=220,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =================================================================
    # PRINT RESULTS
    # =================================================================

    print()

    print(
        "=" * 90
    )

    print(
        "ZERO-SHOT RESAMPLING RESULTS"
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
        "Results:"
    )

    print(
        output_dir
    )


# =====================================================================
# ENTRY
# =====================================================================

if __name__ == "__main__":

    main()