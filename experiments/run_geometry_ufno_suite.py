"""
Generic Geometry-U-FNO diagnostic-suite runner.

This script reuses:

    experiments/plot_final_cfo_vs_geometry_ufno.py

without modifying that existing working file.

It allows us to vary:

    resolution
    Fourier modes
    RK4 steps per segment
    checkpoint

while keeping the same final 01-08 diagnostic suite.

Examples
--------

64x64, modes=12, RK4=2:

python experiments/run_geometry_ufno_suite.py \
    --resolution 64 \
    --modes1 12 \
    --modes2 12 \
    ...

64x64, modes=16:

python experiments/run_geometry_ufno_suite.py \
    --resolution 64 \
    --modes1 16 \
    --modes2 16 \
    ...

64-trained -> 128 zero-shot:

python experiments/run_geometry_ufno_suite.py \
    --resolution 128 \
    --modes1 16 \
    --modes2 16 \
    ...
"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path


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
# EXISTING FINAL SUITE
# =====================================================================

import experiments.plot_final_cfo_vs_geometry_ufno as suite


# =====================================================================
# MODEL IMPORTS
# =====================================================================

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generic Geometry-U-FNO "
            "final diagnostic-suite runner."
        )
    )

    # -----------------------------------------------------------------
    # RESOLUTION
    # -----------------------------------------------------------------

    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--source-resolution",
        type=int,
        default=None,
        help=(
            "Resolution used for training. "
            "Used only for provenance."
        ),
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    parser.add_argument(
        "--id-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ood-data",
        type=str,
        required=True,
    )

    # -----------------------------------------------------------------
    # CFO
    # -----------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    # -----------------------------------------------------------------
    # GEOMETRY-U-FNO CHECKPOINT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--final-ckpt",
        type=str,
        required=True,
    )

    # -----------------------------------------------------------------
    # ARCHITECTURE
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
    # LOSSES
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # PHYSICS / DOMAIN
    # -----------------------------------------------------------------

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--x-min",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--x-max",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--y-min",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--y-max",
        type=float,
        default=2.5,
    )

    # -----------------------------------------------------------------
    # DIAGNOSTICS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
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
        "--direct-case",
        type=str,
        default="hill_right",
    )

    parser.add_argument(
        "--cross-section-case",
        type=str,
        default="hill_center",
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


# =====================================================================
# RESOLVE PATH
# =====================================================================

def resolve_path(
    value,
):

    path = (
        Path(value)
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
# MAIN
# =====================================================================

def main():

    args = parse_args()

    resolution = int(
        args.resolution
    )

    source_resolution = (
        resolution
        if args.source_resolution is None
        else
        int(
            args.source_resolution
        )
    )

    # -----------------------------------------------------------------
    # DOMAIN SPACING
    # -----------------------------------------------------------------

    dx = (
        args.x_max
        -
        args.x_min
    ) / resolution

    dy = (
        args.y_max
        -
        args.y_min
    ) / resolution

    cell_area = (
        dx
        *
        dy
    )

    # -----------------------------------------------------------------
    # PATCH RESOLUTION GLOBALS IN EXISTING SUITE
    # -----------------------------------------------------------------

    suite.RESOLUTION = (
        resolution
    )

    suite.X_MIN = (
        args.x_min
    )

    suite.X_MAX = (
        args.x_max
    )

    suite.Y_MIN = (
        args.y_min
    )

    suite.Y_MAX = (
        args.y_max
    )

    suite.DX = (
        dx
    )

    suite.DY = (
        dy
    )

    suite.CELL_AREA = (
        cell_area
    )

    suite.GRAVITY = (
        args.gravity
    )

    # -----------------------------------------------------------------
    # REPLACE ONLY THE MODEL-BUILDER FUNCTION
    #
    # This allows the existing plotting suite to use modes=12 OR 16,
    # without editing the original large plotting file.
    # -----------------------------------------------------------------

    original_builder = (
        suite.build_final_method
    )

    def build_final_method_dynamic(
        inner_args,
    ):

        model = (
            GeometryUFNO2d(
                num_channels=3,

                modes1=(
                    args.modes1
                ),

                modes2=(
                    args.modes2
                ),

                width=(
                    args.width
                ),

                num_blocks=(
                    args.num_blocks
                ),

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

                include_gradient_magnitude=False,

                use_time=True,
            )
        )

        return (
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

                dx=dx,

                dy=dy,

                gravity=(
                    args.gravity
                ),
            )
        )

    suite.build_final_method = (
        build_final_method_dynamic
    )

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
    # PROVENANCE
    # -----------------------------------------------------------------

    provenance_path = (
        output_dir
        /
        "experiment_configuration.txt"
    )

    with open(
        provenance_path,
        "w",
    ) as file:

        file.write(
            "Geometry-U-FNO diagnostic configuration\n"
        )

        file.write(
            "=====================================\n"
        )

        file.write(
            f"source_resolution = "
            f"{source_resolution}\n"
        )

        file.write(
            f"evaluation_resolution = "
            f"{resolution}\n"
        )

        file.write(
            f"dx = {dx}\n"
        )

        file.write(
            f"dy = {dy}\n"
        )

        file.write(
            f"modes1 = {args.modes1}\n"
        )

        file.write(
            f"modes2 = {args.modes2}\n"
        )

        file.write(
            f"width = {args.width}\n"
        )

        file.write(
            f"num_blocks = "
            f"{args.num_blocks}\n"
        )

        file.write(
            f"num_u_blocks = "
            f"{args.num_u_blocks}\n"
        )

        file.write(
            f"geometry_width = "
            f"{args.geometry_width}\n"
        )

        file.write(
            f"geometry_depth = "
            f"{args.geometry_depth}\n"
        )

        file.write(
            f"lambda_pde = "
            f"{args.lambda_pde}\n"
        )

        file.write(
            f"lambda_bed = "
            f"{args.lambda_bed}\n"
        )

        file.write(
            f"lambda_wb = "
            f"{args.lambda_wb}\n"
        )

        file.write(
            f"steps_per_segment = "
            f"{args.steps_per_segment}\n"
        )

        file.write(
            f"final_checkpoint = "
            f"{resolve_path(args.final_ckpt)}\n"
        )

    # -----------------------------------------------------------------
    # FORWARD ONLY ARGUMENTS UNDERSTOOD BY ORIGINAL SUITE
    # -----------------------------------------------------------------

    forwarded = [

        "plot_final_cfo_vs_geometry_ufno.py",

        "--id-data",
        str(
            resolve_path(
                args.id_data
            )
        ),

        "--counterfactual-data",
        str(
            resolve_path(
                args.counterfactual_data
            )
        ),

        "--ood-data",
        str(
            resolve_path(
                args.ood_data
            )
        ),

        "--cfo-ckpt-dir",
        str(
            resolve_path(
                args.cfo_ckpt_dir
            )
        ),

        "--cfo-prefix",
        args.cfo_prefix,

        "--final-ckpt",
        str(
            resolve_path(
                args.final_ckpt
            )
        ),

        "--lambda-pde",
        str(
            args.lambda_pde
        ),

        "--lambda-bed",
        str(
            args.lambda_bed
        ),

        "--lambda-wb",
        str(
            args.lambda_wb
        ),

        "--wb-eta0",
        str(
            args.wb_eta0
        ),

        "--plot-time",
        str(
            args.plot_time
        ),

        "--steps-per-segment",
        str(
            args.steps_per_segment
        ),

        "--sample-index",
        str(
            args.sample_index
        ),

        "--direct-case",
        args.direct_case,

        "--cross-section-case",
        args.cross_section_case,

        "--output-dir",
        str(
            output_dir
        ),
    ]

    print()
    print(
        "=" * 80
    )

    print(
        "GENERIC GEOMETRY-U-FNO DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print(
        "Source resolution:",
        source_resolution,
    )

    print(
        "Evaluation resolution:",
        resolution,
    )

    print(
        "Fourier modes:",
        (
            args.modes1,
            args.modes2,
        ),
    )

    print(
        "RK4 steps per segment:",
        args.steps_per_segment,
    )

    print(
        "lambda PDE:",
        args.lambda_pde,
    )

    print(
        "lambda bed:",
        args.lambda_bed,
    )

    print(
        "lambda WB:",
        args.lambda_wb,
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "=" * 80
    )

    old_argv = list(
        sys.argv
    )

    try:

        sys.argv = (
            forwarded
        )

        suite.main()

    finally:

        sys.argv = (
            old_argv
        )

        suite.build_final_method = (
            original_builder
        )

    print()
    print(
        "Configuration saved to:"
    )

    print(
        provenance_path
    )


if __name__ == "__main__":

    main()