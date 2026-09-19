"""
Compare 64x64 Geometry-U-FNO with 12 and 16 Fourier modes.

Comparison
----------
PyClaw 64x64 reference
Geo-U-FNO 64-trained, 12 modes
Geo-U-FNO 64-trained, 16 modes

Outputs
-------
1. 01_direct_condition_test.png
2. 02_isolated_hill_effect_error.png
3. 02_isolated_hill_effect_error.csv
4. 03_terrain_induced_speed_gaussian.png
5. 04_terrain_induced_speed_unseen.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# PROJECT ROOT
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
    "modes12",
    "modes16",
]


MODEL_LABELS = {
    "modes12":
        "Geo-U-FNO 64 / 12 modes",

    "modes16":
        "Geo-U-FNO 64 / 16 modes",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare 12-mode and 16-mode "
            "Geometry-U-FNO at 64x64."
        )
    )

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
        "--modes12-checkpoint",
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
        "--modes16-checkpoint",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res64_modes16/"
            "lamwb_0p1/"
            "seed0"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "geometry_ufno_64_modes12_vs_modes16"
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
        choices=[
            "hill_left",
            "hill_center",
            "hill_right",
        ],
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
# BUILD U-FNO
# =====================================================================

def build_method(
    modes,
):

    model = GeometryUFNO2d(
        num_channels=3,

        modes1=modes,
        modes2=modes,

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
# RESTORE
# =====================================================================

def restore_checkpoint(
    method,
    checkpoint_root,
):

    checkpoint_root = resolve_path(
        checkpoint_root
    )

    if not checkpoint_root.exists():

        raise FileNotFoundError(
            f"Checkpoint root does not exist:\n"
            f"{checkpoint_root}"
        )

    target_state = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,

        beta2=0.99,
    )

    state = load_train_state(
        target_state,

        ckpt_dir=str(
            checkpoint_root
        ),

        prefix="best",

        step=None,

        max_to_keep=1,
    )

    return state


# =====================================================================
# MODEL PAIR EVALUATION
# =====================================================================

def evaluate_case(
    archive,
    case_name,
    *,
    methods,
    states,
    sample_index,
    steps_per_segment,
):

    data = archive.get_pair(
        case_name,
        sample_index=sample_index,
    )

    result = {
        "data":
            data,
    }

    for key in MODEL_KEYS:

        print(
            f"Running "
            f"{MODEL_LABELS[key]} "
            f"for {case_name}"
        )

        (
            hill_prediction,
            flat_prediction,
        ) = diag.model_pair_rollout(
            methods[
                key
            ],

            states[
                key
            ],

            data,

            steps_per_segment=(
                steps_per_segment
            ),
        )

        result[
            key
        ] = {
            "hill":
                hill_prediction,

            "flat":
                flat_prediction,
        }

    return result


# =====================================================================
# DIRECT CONDITION TEST
# =====================================================================

def plot_direct_condition(
    result,
    *,
    methods,
    states,
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

    b_zero = np.zeros_like(
        b
    )

    # -----------------------------------------------------------------
    # SWE EXPECTED BED RESPONSE
    # -----------------------------------------------------------------

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

    responses = {}

    for key in MODEL_KEYS:

        vector_with_bed = (
            diag.model_vector_field(
                methods[
                    key
                ],

                states[
                    key
                ],

                q,

                b,

                time_value,
            )
        )

        vector_flat = (
            diag.model_vector_field(
                methods[
                    key
                ],

                states[
                    key
                ],

                q,

                b_zero,

                time_value,
            )
        )

        responses[
            key
        ] = (
            vector_with_bed
            -
            vector_flat
        )

    hu_fields = [
        expected_hu,

        responses[
            "modes12"
        ][
            ...,
            1
        ],

        responses[
            "modes16"
        ][
            ...,
            1
        ],
    ]

    hv_fields = [
        expected_hv,

        responses[
            "modes12"
        ][
            ...,
            2
        ],

        responses[
            "modes16"
        ][
            ...,
            2
        ],
    ]

    labels = [
        "SWE expected",

        MODEL_LABELS[
            "modes12"
        ],

        MODEL_LABELS[
            "modes16"
        ],
    ]

    hu_lim = max(
        max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            )
            for field
            in hu_fields
        ),
        1.0e-8,
    )

    hv_lim = max(
        max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            )
            for field
            in hv_fields
        ),
        1.0e-8,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            10,
        ),
    )

    for column in range(
        3
    ):

        image = diag.heatmap(
            axes[
                0,
                column
            ],

            hu_fields[
                column
            ],

            vlim=hu_lim,

            title=(
                labels[
                    column
                ]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),

            b=b,
        )

        fig.colorbar(
            image,

            ax=axes[
                0,
                column
            ],

            shrink=0.82,
        )

        image = diag.heatmap(
            axes[
                1,
                column
            ],

            hv_fields[
                column
            ],

            vlim=hv_lim,

            title=(
                labels[
                    column
                ]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),

            b=b,
        )

        fig.colorbar(
            image,

            ax=axes[
                1,
                column
            ],

            shrink=0.82,
        )

    fig.suptitle(
        (
            "64x64 U-FNO spectral-mode comparison\n"
            f"Direct condition test at "
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
# ISOLATED HILL EFFECT
# =====================================================================

def plot_isolated_hill_error(
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

    rows = []

    for column, case_name in enumerate(
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
                column
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

            for index in range(
                len(
                    time
                )
            ):

                rows.append(
                    {
                        "case":
                            case_name,

                        "time":
                            float(
                                time[
                                    index
                                ]
                            ),

                        "model":
                            MODEL_LABELS[
                                key
                            ],

                        "relative_effect_error":
                            float(
                                error[
                                    index
                                ]
                            ),
                    }
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
            (
                "Relative error in "
                "terrain-induced Δq"
            )
        )

        axes[
            column
        ].grid(
            alpha=0.30
        )

        axes[
            column
        ].legend()

    fig.suptitle(
        (
            "64x64 isolated Gaussian-hill "
            "effect: 12 vs 16 Fourier modes"
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
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,

            fieldnames=[
                "case",
                "time",
                "model",
                "relative_effect_error",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# =====================================================================
# TERRAIN SPEED
# =====================================================================

def plot_speed_grid(
    results,
    *,
    case_names,
    plot_time,
    output_path,
    title,
):

    number_rows = len(
        case_names
    )

    fig, axes = plt.subplots(
        number_rows,
        3,
        figsize=(
            15,
            4.7
            *
            number_rows,
        ),
    )

    if number_rows == 1:

        axes = np.asarray(
            axes
        )[
            None,
            :
        ]

    labels = [
        "PyClaw 64",

        MODEL_LABELS[
            "modes12"
        ],

        MODEL_LABELS[
            "modes16"
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
                    "modes12"
                ][
                    "hill"
                ],

                result[
                    "modes12"
                ][
                    "flat"
                ],

                index,
            ),

            diag.terrain_speed_effect(
                result[
                    "modes16"
                ][
                    "hill"
                ],

                result[
                    "modes16"
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

        for column in range(
            3
        ):

            image = diag.heatmap(
                axes[
                    row,
                    column
                ],

                fields[
                    column
                ],

                vlim=vlim,

                title=(
                    case_name
                    +
                    "\n"
                    +
                    labels[
                        column
                    ]
                ),

                b=data[
                    "b"
                ],
            )

            fig.colorbar(
                image,

                ax=axes[
                    row,
                    column
                ],

                shrink=0.80,

                label=(
                    r"$|\mathbf{u}|_{terrain}"
                    r"-|\mathbf{u}|_{flat}$"
                ),
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
    # EXISTING DIAGNOSTIC HELPERS WERE WRITTEN FOR 32x32.
    # Update their grid globals to 64x64.
    # -----------------------------------------------------------------

    diag.NX = RESOLUTION
    diag.NY = RESOLUTION

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = X_MIN
    diag.X_MAX = X_MAX

    diag.Y_MIN = Y_MIN
    diag.Y_MAX = Y_MAX

    # -----------------------------------------------------------------
    # BUILD BOTH MODELS
    # -----------------------------------------------------------------

    methods = {
        "modes12":
            build_method(
                12
            ),

        "modes16":
            build_method(
                16
            ),
    }

    # -----------------------------------------------------------------
    # RESTORE
    # -----------------------------------------------------------------

    print()
    print(
        "Restoring 12-mode 64x64 model..."
    )

    state12 = restore_checkpoint(
        methods[
            "modes12"
        ],

        args.modes12_checkpoint,
    )

    print(
        "12-mode model restored."
    )

    print()
    print(
        "Restoring 16-mode 64x64 model..."
    )

    state16 = restore_checkpoint(
        methods[
            "modes16"
        ],

        args.modes16_checkpoint,
    )

    print(
        "16-mode model restored."
    )

    states = {
        "modes12":
            state12,

        "modes16":
            state16,
    }

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    gaussian_archive = diag.H5Archive(
        args.counterfactual_data
    )

    ood_archive = diag.H5Archive(
        args.ood_data
    )

    try:

        # =============================================================
        # GAUSSIAN
        # =============================================================

        gaussian_results = {}

        for case_name in diag.GAUSSIAN_CASES:

            gaussian_results[
                case_name
            ] = evaluate_case(
                gaussian_archive,

                case_name,

                methods=methods,

                states=states,

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # =============================================================
        # OOD
        # =============================================================

        ood_results = {}

        for case_name in diag.OOD_CASES:

            ood_results[
                case_name
            ] = evaluate_case(
                ood_archive,

                case_name,

                methods=methods,

                states=states,

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # =============================================================
        # DIRECT CONDITION
        # =============================================================

        plot_direct_condition(
            gaussian_results[
                args.direct_case
            ],

            methods=methods,

            states=states,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                output_dir
                /
                "01_direct_condition_test.png"
            ),
        )

        # =============================================================
        # ISOLATED HILL
        # =============================================================

        plot_isolated_hill_error(
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

        # =============================================================
        # GAUSSIAN SPEED
        # =============================================================

        plot_speed_grid(
            gaussian_results,

            case_names=(
                diag.GAUSSIAN_CASES
            ),

            plot_time=(
                args.plot_time
            ),

            output_path=(
                output_dir
                /
                "03_terrain_induced_speed_gaussian.png"
            ),

            title=(
                "64x64 terrain-induced speed: "
                "12 vs 16 Fourier modes"
            ),
        )

        # =============================================================
        # OOD SPEED
        # =============================================================

        plot_speed_grid(
            ood_results,

            case_names=(
                diag.OOD_CASES
            ),

            plot_time=(
                args.plot_time
            ),

            output_path=(
                output_dir
                /
                "04_terrain_induced_speed_unseen.png"
            ),

            title=(
                "64x64 unseen-terrain speed: "
                "12 vs 16 Fourier modes"
            ),
        )

        print()
        print(
            "=" * 72
        )

        print(
            "12-VS-16 MODE DIAGNOSTICS COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            "Saved to:"
        )

        print(
            output_dir
        )

    finally:

        gaussian_archive.close()

        ood_archive.close()


if __name__ == "__main__":
    main()