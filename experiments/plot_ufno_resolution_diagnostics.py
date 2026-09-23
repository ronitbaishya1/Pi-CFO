"""
Resolution diagnostics for Geometry-U-FNO.

Compare:

    PyClaw 64x64 reference
    32x32-trained Geometry-U-FNO evaluated zero-shot at 64x64
    64x64-trained Geometry-U-FNO

Outputs
-------
1. 01_direct_condition_test.png
2. 02_isolated_hill_effect_error.png
3. 03_terrain_induced_speed_gaussian.png
4. 04_terrain_induced_speed_unseen.png
5. 02_isolated_hill_effect_error.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
# PROJECT IMPORTS
# =====================================================================

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

GRAVITY = 1.0


MODEL_KEYS = [
    "zero_shot",
    "trained64",
]


MODEL_LABELS = {
    "zero_shot":
        "Geo-U-FNO 32→64",

    "trained64":
        "Geo-U-FNO 64-trained",
}


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
        "--ood-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_64.h5"
        ),
    )

    parser.add_argument(
        "--zero-shot-checkpoint-root",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "lamwb_0p1/"
            "seed0"
        ),
    )

    parser.add_argument(
        "--trained64-checkpoint-root",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res64/"
            "lamwb_0p1/"
            "seed0"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "geometry_ufno_resolution_diagnostics"
        ),
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
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
        "--direct-case",
        type=str,
        default="hill_right",
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
# BUILD METHOD
# =====================================================================

def build_method():

    model = GeometryUFNO2d(
        num_channels=3,

        modes1=12,
        modes2=12,

        width=64,
        num_blocks=4,

        num_u_blocks=2,

        geometry_width=16,
        geometry_depth=2,

        dx=DX,
        dy=DY,

        include_gradient_magnitude=False,

        use_time=True,
    )

    return WellBalancedBathymetryBedPICFO(
        model=model,

        input_shape=(
            RESOLUTION,
            RESOLUTION,
            3,
        ),

        condition_shape=(
            RESOLUTION,
            RESOLUTION,
            1,
        ),

        gamma=1.0e-5,

        spline_type="quintic",

        lambda_pde=0.03,

        lambda_bed=0.70,

        lambda_wb=0.10,

        wb_eta0=1.5,

        dx=DX,

        dy=DY,

        gravity=GRAVITY,
    )


# =====================================================================
# RESTORE CHECKPOINT
# =====================================================================

def restore(
    method,
    checkpoint_root,
):

    checkpoint_root = resolve_path(
        checkpoint_root
    )

    target_state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target_state,

        ckpt_dir=str(
            checkpoint_root
        ),

        prefix="best",

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# ONE CASE
# =====================================================================

def evaluate_case(
    archive,
    case_name,
    *,
    method,
    zero_state,
    trained_state,
    sample_index,
    steps_per_segment,
):

    data = archive.get_pair(
        case_name,
        sample_index=sample_index,
    )

    zero_hill, zero_flat = (
        diag.model_pair_rollout(
            method,
            zero_state,
            data,
            steps_per_segment=(
                steps_per_segment
            ),
        )
    )

    trained_hill, trained_flat = (
        diag.model_pair_rollout(
            method,
            trained_state,
            data,
            steps_per_segment=(
                steps_per_segment
            ),
        )
    )

    return {
        "data":
            data,

        "zero_shot": {
            "hill":
                zero_hill,
            "flat":
                zero_flat,
        },

        "trained64": {
            "hill":
                trained_hill,
            "flat":
                trained_flat,
        },
    }


# =====================================================================
# DIRECT CONDITION TEST
# =====================================================================

def plot_direct_condition(
    result,
    *,
    method,
    zero_state,
    trained_state,
    plot_time,
    output_path,
):

    data = result[
        "data"
    ]

    time = data[
        "time"
    ]

    index = int(
        np.argmin(
            np.abs(
                time
                -
                plot_time
            )
        )
    )

    time_value = float(
        time[
            index
        ]
    )

    q = data[
        "terrain_q"
    ][
        index
    ]

    b = data[
        "b"
    ]

    b0 = np.zeros_like(
        b
    )

    h = np.maximum(
        q[
            ...,
            0
        ],
        1.0e-6,
    )

    expected_hu = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_x(
            b
        )
    )

    expected_hv = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_y(
            b
        )
    )

    states = {
        "zero_shot":
            zero_state,

        "trained64":
            trained_state,
    }

    responses = {}

    for key in MODEL_KEYS:

        with_b = diag.model_vector_field(
            method,
            states[
                key
            ],
            q,
            b,
            time_value,
        )

        flat_b = diag.model_vector_field(
            method,
            states[
                key
            ],
            q,
            b0,
            time_value,
        )

        responses[
            key
        ] = (
            with_b
            -
            flat_b
        )

    hu_fields = [
        expected_hu,
        responses[
            "zero_shot"
        ][
            ...,
            1
        ],
        responses[
            "trained64"
        ][
            ...,
            1
        ],
    ]

    hv_fields = [
        expected_hv,
        responses[
            "zero_shot"
        ][
            ...,
            2
        ],
        responses[
            "trained64"
        ][
            ...,
            2
        ],
    ]

    labels = [
        "SWE expected",
        MODEL_LABELS[
            "zero_shot"
        ],
        MODEL_LABELS[
            "trained64"
        ],
    ]

    hu_lim = max(
        float(
            np.max(
                np.abs(
                    field
                )
            )
        )
        for field
        in hu_fields
    )

    hv_lim = max(
        float(
            np.max(
                np.abs(
                    field
                )
            )
        )
        for field
        in hv_fields
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            10,
        ),
    )

    for col in range(
        3
    ):

        im = diag.heatmap(
            axes[
                0,
                col
            ],

            hu_fields[
                col
            ],

            vlim=hu_lim,

            title=(
                labels[
                    col
                ]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),

            b=b,
        )

        fig.colorbar(
            im,
            ax=axes[
                0,
                col
            ],
        )

        im = diag.heatmap(
            axes[
                1,
                col
            ],

            hv_fields[
                col
            ],

            vlim=hv_lim,

            title=(
                labels[
                    col
                ]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),

            b=b,
        )

        fig.colorbar(
            im,
            ax=axes[
                1,
                col
            ],
        )

    fig.suptitle(
        (
            "64x64 direct-condition resolution test\n"
            f"t = {time_value:.2f}"
        ),
        fontsize=17,
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
# ISOLATED HILL ERROR
# =====================================================================

def plot_hill_error(
    results,
    *,
    output_path,
    csv_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
    )

    csv_rows = []

    for col, case_name in enumerate(
        diag.GAUSSIAN_CASES
    ):

        result = results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        for key in MODEL_KEYS:

            error = (
                diag.relative_effect_error(
                    data[
                        "terrain_q"
                    ],
                    data[
                        "flat_q"
                    ],
                    result[
                        key
                    ][
                        "hill"
                    ],
                    result[
                        key
                    ][
                        "flat"
                    ],
                )
            )

            axes[
                col
            ].plot(
                time,
                error,
                linewidth=2,
                label=(
                    MODEL_LABELS[
                        key
                    ]
                ),
            )

            for i in range(
                len(
                    time
                )
            ):

                csv_rows.append(
                    {
                        "case":
                            case_name,

                        "time":
                            float(
                                time[
                                    i
                                ]
                            ),

                        "model":
                            MODEL_LABELS[
                                key
                            ],

                        "relative_effect_error":
                            float(
                                error[
                                    i
                                ]
                            ),
                    }
                )

        axes[
            col
        ].set_title(
            case_name
        )

        axes[
            col
        ].set_xlabel(
            "Time"
        )

        axes[
            col
        ].set_ylabel(
            "Relative error in terrain-induced Δq"
        )

        axes[
            col
        ].grid(
            alpha=0.3
        )

        axes[
            col
        ].legend()

    fig.suptitle(
        (
            "64x64 isolated Gaussian-hill "
            "resolution comparison"
        ),
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

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case",
                "time",
                "model",
                "relative_effect_error",
            ],
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )


# =====================================================================
# SPEED GRID
# =====================================================================

def plot_speed_grid(
    results,
    *,
    case_names,
    plot_time,
    output_path,
    title,
):

    rows = len(
        case_names
    )

    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(
            15,
            4.7
            *
            rows,
        ),
    )

    labels = [
        "PyClaw 64",
        MODEL_LABELS[
            "zero_shot"
        ],
        MODEL_LABELS[
            "trained64"
        ],
    ]

    for row, case_name in enumerate(
        case_names
    ):

        result = results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        index = int(
            np.argmin(
                np.abs(
                    time
                    -
                    plot_time
                )
            )
        )

        fields = [
            diag.terrain_speed_effect(
                data[
                    "terrain_q"
                ],
                data[
                    "flat_q"
                ],
                index,
            ),

            diag.terrain_speed_effect(
                result[
                    "zero_shot"
                ][
                    "hill"
                ],
                result[
                    "zero_shot"
                ][
                    "flat"
                ],
                index,
            ),

            diag.terrain_speed_effect(
                result[
                    "trained64"
                ][
                    "hill"
                ],
                result[
                    "trained64"
                ][
                    "flat"
                ],
                index,
            ),
        ]

        vlim = max(
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
            1.0e-8,
        )

        for col in range(
            3
        ):

            im = diag.heatmap(
                axes[
                    row,
                    col
                ],

                fields[
                    col
                ],

                vlim=vlim,

                title=(
                    case_name
                    +
                    "\n"
                    +
                    labels[
                        col
                    ]
                ),

                b=data[
                    "b"
                ],
            )

            fig.colorbar(
                im,
                ax=axes[
                    row,
                    col
                ],
                shrink=0.80,
            )

    fig.suptitle(
        (
            title
            +
            f" at t = {plot_time:.2f}"
        ),
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
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
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # Update the existing diagnostic utility to the 64 grid.
    # Its functions read these globals dynamically.
    # -----------------------------------------------------------------

    diag.NX = RESOLUTION
    diag.NY = RESOLUTION

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = X_MIN
    diag.X_MAX = X_MAX

    diag.Y_MIN = Y_MIN
    diag.Y_MAX = Y_MAX

    method = build_method()

    print(
        "Restoring 32-trained U-FNO..."
    )

    zero_state = restore(
        method,
        args.zero_shot_checkpoint_root,
    )

    print(
        "Restoring 64-trained U-FNO..."
    )

    trained_state = restore(
        method,
        args.trained64_checkpoint_root,
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gaussian_archive = diag.H5Archive(
        args.counterfactual_data
    )

    ood_archive = diag.H5Archive(
        args.ood_data
    )

    try:

        gaussian_results = {}

        for case_name in diag.GAUSSIAN_CASES:

            gaussian_results[
                case_name
            ] = evaluate_case(
                gaussian_archive,
                case_name,

                method=method,

                zero_state=(
                    zero_state
                ),

                trained_state=(
                    trained_state
                ),

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        ood_results = {}

        for case_name in diag.OOD_CASES:

            ood_results[
                case_name
            ] = evaluate_case(
                ood_archive,
                case_name,

                method=method,

                zero_state=(
                    zero_state
                ),

                trained_state=(
                    trained_state
                ),

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # -------------------------------------------------------------
        # DIRECT CONDITION
        # -------------------------------------------------------------

        plot_direct_condition(
            gaussian_results[
                args.direct_case
            ],

            method=method,

            zero_state=zero_state,

            trained_state=trained_state,

            plot_time=args.plot_time,

            output_path=(
                output_dir
                /
                "01_direct_condition_test.png"
            ),
        )

        # -------------------------------------------------------------
        # ISOLATED HILL
        # -------------------------------------------------------------

        plot_hill_error(
            gaussian_results,

            output_path=(
                output_dir
                /
                "02_isolated_hill_effect_error.png"
            ),

            csv_path=(
                output_dir
                /
                "02_isolated_hill_effect_error.csv"
            ),
        )

        # -------------------------------------------------------------
        # GAUSSIAN SPEED
        # -------------------------------------------------------------

        plot_speed_grid(
            gaussian_results,

            case_names=(
                diag.GAUSSIAN_CASES
            ),

            plot_time=args.plot_time,

            output_path=(
                output_dir
                /
                "03_terrain_induced_speed_gaussian.png"
            ),

            title=(
                "64x64 terrain-induced speed "
                "for Gaussian hills"
            ),
        )

        # -------------------------------------------------------------
        # OOD SPEED
        # -------------------------------------------------------------

        plot_speed_grid(
            ood_results,

            case_names=(
                diag.OOD_CASES
            ),

            plot_time=args.plot_time,

            output_path=(
                output_dir
                /
                "04_terrain_induced_speed_unseen.png"
            ),

            title=(
                "64x64 terrain-induced speed "
                "on unseen bathymetry"
            ),
        )

        print()
        print(
            "Resolution diagnostics complete."
        )

        print(
            "Saved to:",
            output_dir,
        )

    finally:

        gaussian_archive.close()

        ood_archive.close()


if __name__ == "__main__":
    main()