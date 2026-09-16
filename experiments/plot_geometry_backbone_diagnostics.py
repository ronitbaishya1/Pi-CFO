"""
Compare the four well-balanced PI-CFO architectures:

    1. WB-FNO
    2. Geometry-FNO
    3. Geometry-U-FNO
    4. Geometry-F-FNO

against PyClaw / SWE reference behavior.

Outputs
-------
1. 01_direct_condition_test.png
2. 02_isolated_hill_effect_error.png
3. 03_terrain_induced_speed_gaussian.png
4. 04_terrain_induced_speed_unseen.png
5. 02_isolated_hill_effect_error.csv

Reference datasets
------------------
data/shallow_water_bathy/gaussian_counterfactual_32.h5
data/shallow_water_bathy/swe_bathy_ood_32.h5

Important
---------
The existing experiment:

    experiments/plot_wb_bed_diagnostics.py

is reused for:

    HDF5 reading
    rollout utilities
    vector-field evaluation
    terrain-effect calculations
    heatmap plotting
    checkpoint restoration

This avoids changing the already-tested diagnostic definitions.
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
# MODELS
# =====================================================================

from models.fno import (
    FNO2d,
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


# =====================================================================
# PI-CFO
# =====================================================================

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)


# =====================================================================
# REUSE EXISTING TESTED DIAGNOSTIC UTILITIES
# =====================================================================

from experiments.plot_wb_bed_diagnostics import (
    H5Archive,
    resolve_path,
    restore_checkpoint,
    model_pair_rollout,
    model_vector_field,
    relative_effect_error,
    terrain_speed_effect,
    central_diff_x,
    central_diff_y,
    heatmap,
    GAUSSIAN_CASES,
    OOD_CASES,
    GRAVITY,
)


# =====================================================================
# CONSTANTS
# =====================================================================

NX = 32
NY = 32

DOMAIN_LENGTH_X = 5.0
DOMAIN_LENGTH_Y = 5.0

DX = (
    DOMAIN_LENGTH_X
    /
    NX
)

DY = (
    DOMAIN_LENGTH_Y
    /
    NY
)


# =====================================================================
# MODEL ORDER
# =====================================================================

MODEL_ORDER = [
    "wb_fno",
    "geometry_fno",
    "geometry_ufno",
    "geometry_ffno",
]


MODEL_LABELS = {
    "wb_fno":
        "WB-FNO",

    "geometry_fno":
        "Geometry-FNO",

    "geometry_ufno":
        "Geometry-U-FNO",

    "geometry_ffno":
        "Geometry-F-FNO",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Compare WB-FNO, Geometry-FNO, "
            "Geometry-U-FNO and Geometry-F-FNO."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_32.h5"
        ),
    )

    parser.add_argument(
        "--ood-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_32.h5"
        ),
    )

    # -----------------------------------------------------------------
    # CHECKPOINTS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--wb-fno-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "wb_bathy_bed_pi/"
            "fno/"
            "lamwb_0p1/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--geometry-fno-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_fno/"
            "res32/"
            "lamwb_0p1/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--geometry-ufno-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "lamwb_0p1/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--geometry-ffno-ckpt",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ffno/"
            "res32/"
            "lamwb_0p1/"
            "seed0/"
            "best"
        ),
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "geometry_backbone_diagnostics"
        ),
    )

    # -----------------------------------------------------------------
    # PHYSICS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.03,
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

    # -----------------------------------------------------------------
    # DIAGNOSTICS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--direct-case",
        type=str,
        default="hill_right",
        choices=GAUSSIAN_CASES,
    )

    return parser.parse_args()


# =====================================================================
# COMMON WELL-BALANCED METHOD
# =====================================================================

def build_method(
    model,
    *,
    args,
):

    return WellBalancedBathymetryBedPICFO(
        model=model,

        input_shape=(
            NX,
            NY,
            3,
        ),

        condition_shape=(
            NX,
            NY,
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

        dx=DX,

        dy=DY,

        gravity=GRAVITY,
    )


# =====================================================================
# BUILD ALL MODELS
# =====================================================================

def build_methods(
    args,
):

    # -----------------------------------------------------------------
    # ORIGINAL WB-FNO
    # -----------------------------------------------------------------

    wb_fno_model = FNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        use_condition=True,
        use_time=True,
    )

    # -----------------------------------------------------------------
    # GEOMETRY-FNO
    # -----------------------------------------------------------------

    geometry_fno_model = (
        GeometryFNO2d(
            num_channels=3,
            modes1=12,
            modes2=12,
            width=64,
            num_blocks=4,

            geometry_width=16,
            geometry_depth=2,

            dx=DX,
            dy=DY,

            include_gradient_magnitude=False,

            use_time=True,
        )
    )

    # -----------------------------------------------------------------
    # GEOMETRY-U-FNO
    # -----------------------------------------------------------------

    geometry_ufno_model = (
        GeometryUFNO2d(
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
    )

    # -----------------------------------------------------------------
    # GEOMETRY-F-FNO
    # -----------------------------------------------------------------

    geometry_ffno_model = (
        GeometryFFNO2d(
            num_channels=3,
            modes1=12,
            modes2=12,
            width=64,
            num_blocks=4,

            expansion=2,

            geometry_width=16,
            geometry_depth=2,

            dx=DX,
            dy=DY,

            include_gradient_magnitude=False,

            use_time=True,
        )
    )

    return {

        "wb_fno":
            build_method(
                wb_fno_model,
                args=args,
            ),

        "geometry_fno":
            build_method(
                geometry_fno_model,
                args=args,
            ),

        "geometry_ufno":
            build_method(
                geometry_ufno_model,
                args=args,
            ),

        "geometry_ffno":
            build_method(
                geometry_ffno_model,
                args=args,
            ),
    }


# =====================================================================
# RESTORE ALL CHECKPOINTS
# =====================================================================

def restore_all_models(
    methods,
    args,
):

    checkpoint_paths = {

        "wb_fno":
            args.wb_fno_ckpt,

        "geometry_fno":
            args.geometry_fno_ckpt,

        "geometry_ufno":
            args.geometry_ufno_ckpt,

        "geometry_ffno":
            args.geometry_ffno_ckpt,
    }

    states = {}

    print()
    print(
        "=" * 72
    )

    print(
        "RESTORING MODEL CHECKPOINTS"
    )

    print(
        "=" * 72
    )

    for model_key in MODEL_ORDER:

        print()

        print(
            "Restoring",
            MODEL_LABELS[
                model_key
            ],
        )

        print(
            "Checkpoint:",
            resolve_path(
                checkpoint_paths[
                    model_key
                ]
            ),
        )

        states[
            model_key
        ] = restore_checkpoint(
            methods[
                model_key
            ],
            checkpoint_paths[
                model_key
            ],
        )

        print(
            MODEL_LABELS[
                model_key
            ],
            "restored."
        )

    return states


# =====================================================================
# RUN ALL FOUR MODELS FOR ONE CASE
# =====================================================================

def evaluate_case(
    *,
    archive,
    case_name,
    methods,
    states,
    sample_index,
    steps_per_segment,
):

    data = archive.get_pair(
        case_name,
        sample_index=sample_index,
    )

    model_results = {}

    for model_key in MODEL_ORDER:

        print()

        print(
            "Running",
            MODEL_LABELS[
                model_key
            ],
            ":",
            case_name,
        )

        (
            q_hill,
            q_flat,
        ) = model_pair_rollout(
            methods[
                model_key
            ],
            states[
                model_key
            ],
            data,
            steps_per_segment=(
                steps_per_segment
            ),
        )

        model_results[
            model_key
        ] = {
            "hill":
                q_hill,

            "flat":
                q_flat,
        }

    return {
        "data":
            data,

        "models":
            model_results,
    }


# =====================================================================
# PLOT 1
# DIRECT CONDITION TEST
# =====================================================================

def plot_direct_condition_test(
    *,
    result,
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

    time_index = int(
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
            time_index
        ]
    )

    q_same = (
        data[
            "terrain_q"
        ][
            time_index
        ]
    )

    bathymetry = data[
        "b"
    ]

    zero_bathymetry = (
        np.zeros_like(
            bathymetry
        )
    )

    # -----------------------------------------------------------------
    # SWE EXPECTED DIRECT BED RESPONSE
    # -----------------------------------------------------------------

    h = np.maximum(
        q_same[
            ...,
            0
        ],
        1.0e-6,
    )

    db_dx = central_diff_x(
        bathymetry
    )

    db_dy = central_diff_y(
        bathymetry
    )

    expected_hu = (
        -GRAVITY
        *
        h
        *
        db_dx
    )

    expected_hv = (
        -GRAVITY
        *
        h
        *
        db_dy
    )

    # -----------------------------------------------------------------
    # MODEL DIRECT RESPONSES
    # -----------------------------------------------------------------

    direct_response = {}

    for model_key in MODEL_ORDER:

        with_bathymetry = (
            model_vector_field(
                methods[
                    model_key
                ],
                states[
                    model_key
                ],
                q_same,
                bathymetry,
                time_value,
            )
        )

        without_bathymetry = (
            model_vector_field(
                methods[
                    model_key
                ],
                states[
                    model_key
                ],
                q_same,
                zero_bathymetry,
                time_value,
            )
        )

        direct_response[
            model_key
        ] = (
            with_bathymetry
            -
            without_bathymetry
        )

    # -----------------------------------------------------------------
    # ALL COLUMNS
    # -----------------------------------------------------------------

    column_labels = [
        "SWE expected",
        *[
            MODEL_LABELS[
                key
            ]
            for key
            in MODEL_ORDER
        ],
    ]

    hu_fields = [
        expected_hu,
        *[
            direct_response[
                key
            ][
                ...,
                1
            ]
            for key
            in MODEL_ORDER
        ],
    ]

    hv_fields = [
        expected_hv,
        *[
            direct_response[
                key
            ][
                ...,
                2
            ]
            for key
            in MODEL_ORDER
        ],
    ]

    # -----------------------------------------------------------------
    # COMMON COLOR RANGE ACROSS ALL ARCHITECTURES
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------

    num_columns = len(
        column_labels
    )

    fig, axes = plt.subplots(
        2,
        num_columns,
        figsize=(
            5.0
            *
            num_columns,
            9.5,
        ),
    )

    for column_index in range(
        num_columns
    ):

        hu_image = heatmap(
            axes[
                0,
                column_index
            ],

            hu_fields[
                column_index
            ],

            vlim=hu_lim,

            title=(
                column_labels[
                    column_index
                ]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),

            b=bathymetry,
        )

        fig.colorbar(
            hu_image,
            ax=axes[
                0,
                column_index
            ],
            shrink=0.82,
        )

        hv_image = heatmap(
            axes[
                1,
                column_index
            ],

            hv_fields[
                column_index
            ],

            vlim=hv_lim,

            title=(
                column_labels[
                    column_index
                ]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),

            b=bathymetry,
        )

        fig.colorbar(
            hv_image,
            ax=axes[
                1,
                column_index
            ],
            shrink=0.82,
        )

    fig.suptitle(
        (
            "Direct condition test: "
            "same q and t, change only bathymetry\n"
            f"t = {time_value:.2f}"
        ),
        fontsize=18,
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
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
# PLOT 2
# ISOLATED GAUSSIAN-HILL EFFECT ERROR
# =====================================================================

def plot_isolated_hill_effect_error(
    *,
    gaussian_results,
    output_path,
    csv_path,
):

    fig, axes = plt.subplots(
        1,
        len(
            GAUSSIAN_CASES
        ),
        figsize=(
            19,
            5.8,
        ),
    )

    csv_rows = []

    for column_index, case_name in enumerate(
        GAUSSIAN_CASES
    ):

        result = gaussian_results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        ax = axes[
            column_index
        ]

        for model_key in MODEL_ORDER:

            model_result = (
                result[
                    "models"
                ][
                    model_key
                ]
            )

            error = (
                relative_effect_error(
                    data[
                        "terrain_q"
                    ],

                    data[
                        "flat_q"
                    ],

                    model_result[
                        "hill"
                    ],

                    model_result[
                        "flat"
                    ],
                )
            )

            ax.plot(
                time,
                error,
                linewidth=2,
                label=(
                    MODEL_LABELS[
                        model_key
                    ]
                ),
            )

            for index in range(
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
                                    index
                                ]
                            ),

                        "model":
                            MODEL_LABELS[
                                model_key
                            ],

                        "relative_effect_error":
                            float(
                                error[
                                    index
                                ]
                            ),
                    }
                )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            (
                "Relative error in "
                "terrain-induced Δq"
            )
        )

        ax.grid(
            alpha=0.30
        )

        ax.legend(
            fontsize=9
        )

    fig.suptitle(
        (
            "Error in the isolated "
            "Gaussian-hill effect"
        ),
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
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

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

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
            csv_rows
        )


# =====================================================================
# COMMON TERRAIN-SPEED PLOTTER
# =====================================================================

def plot_terrain_speed_grid(
    *,
    results,
    case_names,
    plot_time,
    output_path,
    title,
):

    num_rows = len(
        case_names
    )

    num_columns = (
        1
        +
        len(
            MODEL_ORDER
        )
    )

    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(
            5.0
            *
            num_columns,
            4.6
            *
            num_rows,
        ),
    )

    if num_rows == 1:

        axes = np.asarray(
            axes
        )[
            None,
            :
        ]

    column_labels = [
        "PyClaw",
        *[
            MODEL_LABELS[
                model_key
            ]
            for model_key
            in MODEL_ORDER
        ],
    ]

    for row_index, case_name in enumerate(
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

        time_index = int(
            np.argmin(
                np.abs(
                    time
                    -
                    plot_time
                )
            )
        )

        # -------------------------------------------------------------
        # PYCLAW REFERENCE
        # -------------------------------------------------------------

        fields = [
            terrain_speed_effect(
                data[
                    "terrain_q"
                ],
                data[
                    "flat_q"
                ],
                time_index,
            )
        ]

        # -------------------------------------------------------------
        # FOUR NEURAL OPERATORS
        # -------------------------------------------------------------

        for model_key in MODEL_ORDER:

            model_result = (
                result[
                    "models"
                ][
                    model_key
                ]
            )

            fields.append(
                terrain_speed_effect(
                    model_result[
                        "hill"
                    ],
                    model_result[
                        "flat"
                    ],
                    time_index,
                )
            )

        # -------------------------------------------------------------
        # SHARED SCALE FOR PYCLAW + ALL FOUR MODELS
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # PLOT ROW
        # -------------------------------------------------------------

        for column_index in range(
            num_columns
        ):

            image = heatmap(
                axes[
                    row_index,
                    column_index
                ],

                fields[
                    column_index
                ],

                vlim=vlim,

                title=(
                    case_name
                    +
                    "\n"
                    +
                    column_labels[
                        column_index
                    ]
                ),

                b=data[
                    "b"
                ],
            )

            fig.colorbar(
                image,

                ax=axes[
                    row_index,
                    column_index
                ],

                shrink=0.78,

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
        fontsize=19,
    )

    plt.tight_layout(
        rect=[
            0.0,
            0.0,
            1.0,
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

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "GEOMETRY BACKBONE DIAGNOSTICS"
    )

    print(
        "=" * 72
    )

    print(
        "lambda_PDE:",
        args.lambda_pde,
    )

    print(
        "lambda_bed:",
        args.lambda_bed,
    )

    print(
        "lambda_WB:",
        args.lambda_wb,
    )

    print(
        "plot time:",
        args.plot_time,
    )

    print(
        "=" * 72
    )

    # =================================================================
    # BUILD METHODS
    # =================================================================

    methods = build_methods(
        args
    )

    # =================================================================
    # RESTORE CHECKPOINTS
    # =================================================================

    states = restore_all_models(
        methods,
        args,
    )

    # =================================================================
    # LOAD DATASETS
    # =================================================================

    gaussian_archive = H5Archive(
        args.counterfactual_data
    )

    ood_archive = H5Archive(
        args.ood_data
    )

    try:

        # =============================================================
        # GAUSSIAN COUNTERFACTUAL CASES
        # =============================================================

        gaussian_results = {}

        for case_name in GAUSSIAN_CASES:

            gaussian_results[
                case_name
            ] = evaluate_case(

                archive=(
                    gaussian_archive
                ),

                case_name=(
                    case_name
                ),

                methods=(
                    methods
                ),

                states=(
                    states
                ),

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # =============================================================
        # OOD TERRAIN CASES
        # =============================================================

        ood_results = {}

        for case_name in OOD_CASES:

            ood_results[
                case_name
            ] = evaluate_case(

                archive=(
                    ood_archive
                ),

                case_name=(
                    case_name
                ),

                methods=(
                    methods
                ),

                states=(
                    states
                ),

                sample_index=(
                    args.sample_index
                ),

                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        # =============================================================
        # PLOT 1
        # =============================================================

        print()
        print(
            "Creating Plot 1:"
        )

        print(
            "Direct condition test..."
        )

        plot_direct_condition_test(

            result=(
                gaussian_results[
                    args.direct_case
                ]
            ),

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
        # PLOT 2
        # =============================================================

        print()
        print(
            "Creating Plot 2:"
        )

        print(
            "Isolated hill effect error..."
        )

        plot_isolated_hill_effect_error(

            gaussian_results=(
                gaussian_results
            ),

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
        # PLOT 3
        # =============================================================

        print()
        print(
            "Creating Plot 3:"
        )

        print(
            "Gaussian terrain-induced speed..."
        )

        plot_terrain_speed_grid(

            results=(
                gaussian_results
            ),

            case_names=(
                GAUSSIAN_CASES
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
                "Terrain-induced speed "
                "for Gaussian hills"
            ),
        )

        # =============================================================
        # PLOT 4
        # =============================================================

        print()
        print(
            "Creating Plot 4:"
        )

        print(
            "Unseen-terrain induced speed..."
        )

        plot_terrain_speed_grid(

            results=(
                ood_results
            ),

            case_names=(
                OOD_CASES
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
                "Terrain-induced speed "
                "on unseen bathymetry"
            ),
        )

        # =============================================================
        # FINAL OUTPUT
        # =============================================================

        print()
        print(
            "=" * 72
        )

        print(
            "DIAGNOSTICS COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            "Results saved to:"
        )

        print(
            output_dir
        )

        print()

        print(
            "Generated:"
        )

        print(
            "  01_direct_condition_test.png"
        )

        print(
            "  02_isolated_hill_effect_error.png"
        )

        print(
            "  02_isolated_hill_effect_error.csv"
        )

        print(
            "  03_terrain_induced_speed_gaussian.png"
        )

        print(
            "  04_terrain_induced_speed_unseen.png"
        )

    finally:

        gaussian_archive.close()

        ood_archive.close()


if __name__ == "__main__":
    main()