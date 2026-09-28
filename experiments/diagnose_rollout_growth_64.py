"""
64x64 rollout-growth diagnostic.

Purpose
-------
Separate:

    1. local one-step model error
    2. accumulated free-rollout error

for the final Geometry-U-FNO model.

The diagnostic evaluates:

    PyClaw q(t_i)
        -> model
        -> q_hat(t_{i+1})

for every interval independently ("teacher forced"),

and compares that with an ordinary recursive rollout.

We also measure high-frequency error and Fourier coherence.

No training is performed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
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


# =====================================================================
# IMPORT EXISTING SPECTRAL DIAGNOSTIC
# =====================================================================

import experiments.diagnose_spatial_spectrum_64 as spec

import experiments.plot_wb_bed_diagnostics as diag


# =====================================================================
# CONSTANTS
# =====================================================================

RESOLUTION = 64

X_MIN = -2.5
X_MAX = 2.5

Y_MIN = -2.5
Y_MAX = 2.5

DX = (
    X_MAX
    -
    X_MIN
) / RESOLUTION

DY = (
    Y_MAX
    -
    Y_MIN
) / RESOLUTION


CHANNEL_NAMES = [
    "h",
    "hu",
    "hv",
]


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--case",
        type=str,
        default="hill_center",
        choices=[
            "hill_left",
            "hill_center",
            "hill_right",
        ],
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--final-ckpt",
        type=str,
        default=None,
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

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "rollout_growth_64"
        ),
    )

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve_path(
    value,
):

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
# SINGLE RK4 INTERVAL
# =====================================================================

def model_step(
    method,
    state,
    q,
    bathymetry,
    t0,
    t1,
    *,
    steps_per_segment,
):

    q_batch = jnp.asarray(
        q[
            None,
            ...
        ],
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        bathymetry[
            None,
            ...,
            None,
        ],
        dtype=jnp.float32,
    )

    internal_steps = max(
        int(
            steps_per_segment
        )
        +
        1,
        2,
    )

    prediction = method.infer_at(
        state,

        q_batch,

        s=float(
            t0
        ),

        t=float(
            t1
        ),

        steps=internal_steps,

        condition=condition,

        method="RK4",
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if (
        prediction.ndim == 4
        and
        prediction.shape[0] == 1
    ):

        prediction = prediction[
            0
        ]

    if prediction.ndim != 3:

        raise ValueError(
            "Unexpected one-step shape: "
            f"{prediction.shape}"
        )

    return prediction


# =====================================================================
# TEACHER-FORCED ROLLOUT
# =====================================================================

def teacher_forced_predictions(
    method,
    state,
    truth,
    bathymetry,
    time,
    *,
    steps_per_segment,
):

    predictions = [
        np.asarray(
            truth[
                0
            ],
            dtype=np.float64,
        )
    ]

    for index in range(
        len(
            time
        )
        -
        1
    ):

        prediction = model_step(
            method,
            state,

            truth[
                index
            ],

            bathymetry,

            time[
                index
            ],

            time[
                index
                +
                1
            ],

            steps_per_segment=(
                steps_per_segment
            ),
        )

        predictions.append(
            prediction
        )

    return np.stack(
        predictions,
        axis=0,
    )


# =====================================================================
# FREE ROLLOUT
# =====================================================================

def free_rollout(
    method,
    state,
    q0,
    bathymetry,
    time,
    *,
    steps_per_segment,
):

    q = np.asarray(
        q0,
        dtype=np.float64,
    )

    predictions = [
        q.copy()
    ]

    for index in range(
        len(
            time
        )
        -
        1
    ):

        q = model_step(
            method,
            state,

            q,

            bathymetry,

            time[
                index
            ],

            time[
                index
                +
                1
            ],

            steps_per_segment=(
                steps_per_segment
            ),
        )

        predictions.append(
            q
        )

    return np.stack(
        predictions,
        axis=0,
    )


# =====================================================================
# RELATIVE L2
# =====================================================================

def relative_l2(
    truth,
    prediction,
    eps=1.0e-12,
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
            eps,
        )
    )


# =====================================================================
# FOURIER TRANSFORM
# =====================================================================

def fft_field(
    field,
):

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    centered = (
        field
        -
        np.mean(
            field
        )
    )

    window = spec.hann_window(
        field.shape
    )

    return np.fft.fftshift(
        np.fft.fft2(
            centered
            *
            window
        )
    )


# =====================================================================
# HIGH-FREQUENCY RELATIVE ERROR
# =====================================================================

def high_frequency_relative_error(
    truth,
    prediction,
    eps=1.0e-14,
):

    truth_fft = fft_field(
        truth
    )

    prediction_fft = fft_field(
        prediction
    )

    rho = spec.normalized_frequency_radius(
        truth.shape,
        DX,
        DY,
    )

    mask = (
        rho
        >=
        0.66
    )

    error_fft = (
        prediction_fft
        -
        truth_fft
    )

    numerator = np.sqrt(
        np.sum(
            np.abs(
                error_fft[
                    mask
                ]
            )
            ** 2
        )
    )

    denominator = np.sqrt(
        np.sum(
            np.abs(
                truth_fft[
                    mask
                ]
            )
            ** 2
        )
    )

    return float(
        numerator
        /
        max(
            denominator,
            eps,
        )
    )


# =====================================================================
# HIGH-FREQUENCY COHERENCE
# =====================================================================

def high_frequency_coherence(
    truth,
    prediction,
    eps=1.0e-14,
):

    truth_fft = fft_field(
        truth
    )

    prediction_fft = fft_field(
        prediction
    )

    rho = spec.normalized_frequency_radius(
        truth.shape,
        DX,
        DY,
    )

    mask = (
        rho
        >=
        0.66
    )

    truth_high = (
        truth_fft[
            mask
        ]
    )

    prediction_high = (
        prediction_fft[
            mask
        ]
    )

    numerator = np.abs(
        np.sum(
            prediction_high
            *
            np.conj(
                truth_high
            )
        )
    )

    denominator = np.sqrt(
        np.sum(
            np.abs(
                prediction_high
            )
            ** 2
        )
        *
        np.sum(
            np.abs(
                truth_high
            )
            ** 2
        )
    )

    return float(
        numerator
        /
        max(
            denominator,
            eps,
        )
    )


# =====================================================================
# METRICS BY TIME
# =====================================================================

def build_time_metrics(
    truth,
    prediction,
    time,
    mode_name,
):

    rows = []

    for time_index in range(
        len(
            time
        )
    ):

        for channel_index, channel_name in enumerate(
            CHANNEL_NAMES
        ):

            true_field = (
                truth[
                    time_index,
                    ...,
                    channel_index
                ]
            )

            pred_field = (
                prediction[
                    time_index,
                    ...,
                    channel_index
                ]
            )

            rows.append(
                {
                    "time":
                        float(
                            time[
                                time_index
                            ]
                        ),

                    "mode":
                        mode_name,

                    "channel":
                        channel_name,

                    "relative_l2":
                        relative_l2(
                            true_field,
                            pred_field,
                        ),

                    "high_frequency_relative_error":
                        high_frequency_relative_error(
                            true_field,
                            pred_field,
                        ),

                    "high_frequency_coherence":
                        high_frequency_coherence(
                            true_field,
                            pred_field,
                        ),
                }
            )

    return rows


# =====================================================================
# TERRAIN-EFFECT METRICS
# =====================================================================

def build_terrain_metrics(
    terrain_truth,
    flat_truth,
    terrain_prediction,
    flat_prediction,
    time,
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

    rows = []

    for time_index in range(
        len(
            time
        )
    ):

        for channel_index, channel_name in enumerate(
            CHANNEL_NAMES
        ):

            true_field = (
                true_effect[
                    time_index,
                    ...,
                    channel_index
                ]
            )

            pred_field = (
                predicted_effect[
                    time_index,
                    ...,
                    channel_index
                ]
            )

            rows.append(
                {
                    "time":
                        float(
                            time[
                                time_index
                            ]
                        ),

                    "channel":
                        channel_name,

                    "relative_l2":
                        relative_l2(
                            true_field,
                            pred_field,
                        ),

                    "high_frequency_relative_error":
                        high_frequency_relative_error(
                            true_field,
                            pred_field,
                        ),

                    "high_frequency_coherence":
                        high_frequency_coherence(
                            true_field,
                            pred_field,
                        ),
                }
            )

    return rows


# =====================================================================
# PLOT RELATIVE L2
# =====================================================================

def plot_relative_l2(
    rows,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5,
        ),
    )

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        ax = axes[
            channel_index
        ]

        for mode in [
            "teacher",
            "free",
        ]:

            selected = [
                row
                for row
                in rows
                if (
                    row[
                        "channel"
                    ]
                    ==
                    channel_name
                    and
                    row[
                        "mode"
                    ]
                    ==
                    mode
                )
            ]

            ax.plot(
                [
                    row[
                        "time"
                    ]
                    for row
                    in selected
                ],

                [
                    row[
                        "relative_l2"
                    ]
                    for row
                    in selected
                ],

                linewidth=2,

                label=(
                    "one-step"
                    if
                    mode == "teacher"
                    else
                    "free rollout"
                ),
            )

        ax.set_title(
            channel_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Relative L2"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

    fig.suptitle(
        "64x64 local vs accumulated rollout error",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# PLOT HIGH FREQUENCY ERROR
# =====================================================================

def plot_high_frequency_error(
    rows,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5,
        ),
    )

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        ax = axes[
            channel_index
        ]

        for mode in [
            "teacher",
            "free",
        ]:

            selected = [
                row
                for row
                in rows
                if (
                    row[
                        "channel"
                    ]
                    ==
                    channel_name
                    and
                    row[
                        "mode"
                    ]
                    ==
                    mode
                )
            ]

            ax.plot(
                [
                    row[
                        "time"
                    ]
                    for row
                    in selected
                ],

                [
                    row[
                        "high_frequency_relative_error"
                    ]
                    for row
                    in selected
                ],

                linewidth=2,

                label=(
                    "one-step"
                    if
                    mode == "teacher"
                    else
                    "free rollout"
                ),
            )

        ax.set_title(
            channel_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "High-frequency relative error"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

    fig.suptitle(
        "64x64 high-frequency error accumulation",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# PLOT COHERENCE
# =====================================================================

def plot_coherence(
    rows,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5,
        ),
    )

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        ax = axes[
            channel_index
        ]

        for mode in [
            "teacher",
            "free",
        ]:

            selected = [
                row
                for row
                in rows
                if (
                    row[
                        "channel"
                    ]
                    ==
                    channel_name
                    and
                    row[
                        "mode"
                    ]
                    ==
                    mode
                )
            ]

            ax.plot(
                [
                    row[
                        "time"
                    ]
                    for row
                    in selected
                ],

                [
                    row[
                        "high_frequency_coherence"
                    ]
                    for row
                    in selected
                ],

                linewidth=2,

                label=(
                    "one-step"
                    if
                    mode == "teacher"
                    else
                    "free rollout"
                ),
            )

        ax.set_ylim(
            0.0,
            1.05,
        )

        ax.set_title(
            channel_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "High-frequency Fourier coherence"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

    fig.suptitle(
        "64x64 high-frequency phase/coherence",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# TERRAIN EFFECT
# =====================================================================

def plot_terrain_metrics(
    rows,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5,
        ),
    )

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):

        selected = [
            row
            for row
            in rows
            if (
                row[
                    "channel"
                ]
                ==
                channel_name
            )
        ]

        axes[
            channel_index
        ].plot(
            [
                row[
                    "time"
                ]
                for row
                in selected
            ],

            [
                row[
                    "high_frequency_relative_error"
                ]
                for row
                in selected
            ],

            linewidth=2,

            label="HF error",
        )

        axes[
            channel_index
        ].set_title(
            channel_name
        )

        axes[
            channel_index
        ].set_xlabel(
            "Time"
        )

        axes[
            channel_index
        ].set_ylabel(
            "Terrain-effect HF relative error"
        )

        axes[
            channel_index
        ].grid(
            alpha=0.3
        )

    fig.suptitle(
        "High-frequency error in terrain-induced response",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# SAVE CSV
# =====================================================================

def save_csv(
    rows,
    output_path,
):

    fieldnames = [
        "time",
        "mode",
        "channel",
        "relative_l2",
        "high_frequency_relative_error",
        "high_frequency_coherence",
    ]

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    spec.configure_diag()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # FINAL MODEL
    # -----------------------------------------------------------------

    method = spec.build_final_method(
        args
    )

    state = spec.restore_final_state(
        method,
        args,
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    archive = diag.H5Archive(
        args.counterfactual_data
    )

    try:

        data = archive.get_pair(
            args.case,

            sample_index=(
                args.sample_index
            ),
        )

        terrain_truth = np.asarray(
            data[
                "terrain_q"
            ],
            dtype=np.float64,
        )

        flat_truth = np.asarray(
            data[
                "flat_q"
            ],
            dtype=np.float64,
        )

        bathymetry = np.asarray(
            data[
                "b"
            ],
            dtype=np.float64,
        )

        time = np.asarray(
            data[
                "time"
            ],
            dtype=np.float64,
        )

        # =============================================================
        # TERRAIN — TEACHER FORCED
        # =============================================================

        print()
        print(
            "Teacher-forced terrain prediction..."
        )

        terrain_teacher = (
            teacher_forced_predictions(
                method,
                state,

                terrain_truth,
                bathymetry,
                time,

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )
        )

        # =============================================================
        # TERRAIN — FREE ROLLOUT
        # =============================================================

        print(
            "Free terrain rollout..."
        )

        terrain_free = (
            free_rollout(
                method,
                state,

                terrain_truth[
                    0
                ],

                bathymetry,
                time,

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )
        )

        # =============================================================
        # FLAT — FREE ROLLOUT
        # =============================================================

        print(
            "Free flat rollout..."
        )

        flat_free = (
            free_rollout(
                method,
                state,

                flat_truth[
                    0
                ],

                np.zeros_like(
                    bathymetry
                ),

                time,

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )
        )

        # =============================================================
        # BUILD METRICS
        # =============================================================

        teacher_rows = (
            build_time_metrics(
                terrain_truth,
                terrain_teacher,
                time,
                "teacher",
            )
        )

        free_rows = (
            build_time_metrics(
                terrain_truth,
                terrain_free,
                time,
                "free",
            )
        )

        all_rows = (
            teacher_rows
            +
            free_rows
        )

        terrain_rows = (
            build_terrain_metrics(
                terrain_truth,
                flat_truth,

                terrain_free,
                flat_free,

                time,
            )
        )

        # =============================================================
        # SAVE
        # =============================================================

        save_csv(
            all_rows,

            output_dir
            /
            "rollout_growth_metrics.csv",
        )

        plot_relative_l2(
            all_rows,

            output_dir
            /
            "01_local_vs_free_relative_l2.png",
        )

        plot_high_frequency_error(
            all_rows,

            output_dir
            /
            "02_high_frequency_error_growth.png",
        )

        plot_coherence(
            all_rows,

            output_dir
            /
            "03_high_frequency_coherence.png",
        )

        plot_terrain_metrics(
            terrain_rows,

            output_dir
            /
            "04_terrain_effect_hf_error.png",
        )

        # =============================================================
        # SIMPLE TERMINAL SUMMARY
        # =============================================================

        print()
        print(
            "=" * 72
        )

        print(
            "FINAL-TIME SUMMARY"
        )

        print(
            "=" * 72
        )

        for channel_name in CHANNEL_NAMES:

            teacher_final = [
                row
                for row
                in teacher_rows
                if row[
                    "channel"
                ]
                ==
                channel_name
            ][
                -1
            ]

            free_final = [
                row
                for row
                in free_rows
                if row[
                    "channel"
                ]
                ==
                channel_name
            ][
                -1
            ]

            print()
            print(
                channel_name
            )

            print(
                "  one-step Rel-L2:",
                f"{teacher_final['relative_l2']:.6f}"
            )

            print(
                "  free Rel-L2:    ",
                f"{free_final['relative_l2']:.6f}"
            )

            print(
                "  one-step HF err:",
                f"{teacher_final['high_frequency_relative_error']:.6f}"
            )

            print(
                "  free HF err:    ",
                f"{free_final['high_frequency_relative_error']:.6f}"
            )

            print(
                "  one-step coh.:  ",
                f"{teacher_final['high_frequency_coherence']:.6f}"
            )

            print(
                "  free coherence: ",
                f"{free_final['high_frequency_coherence']:.6f}"
            )

        print()
        print(
            "Saved:"
        )

        print(
            output_dir
        )

    finally:

        archive.close()


if __name__ == "__main__":
    main()