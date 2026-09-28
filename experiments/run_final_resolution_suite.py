"""
Run the final PyClaw vs Bathy-CFO vs Geometry-U-FNO diagnostic suite
at an arbitrary square spatial resolution.

This is a thin wrapper around:

    experiments/plot_final_cfo_vs_geometry_ufno.py

The underlying diagnostic script already produces:

Tables
------
table_00_model_configuration
table_01_id_metrics
table_02_gaussian_rollout_metrics
table_03_gaussian_mass_drift
table_04_gaussian_terrain_effect
table_05_direct_condition_metrics
table_06_ood_rollout_metrics
table_07_ood_terrain_effect
table_08_final_summary

Figures
-------
01_direct_condition_test.png
02_isolated_hill_effect_error.png
03_terrain_induced_speed_gaussian.png
04_terrain_induced_speed_unseen.png
05_gaussian_cross_sections_h_hu_hv.png
06_gaussian_3d_bathymetry_cross_sections.png
07_gaussian_3d_free_surface.png
08_gaussian_3d_h_hu_hv_cross_sections.png

This wrapper only changes resolution-dependent constants and forwards
the requested data/checkpoint paths.

It can therefore be used for:

    64-trained -> 64 evaluation

or

    64-trained -> 128 zero-shot evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py


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
# IMPORT EXISTING FINAL DIAGNOSTIC SUITE
# =====================================================================

import experiments.plot_final_cfo_vs_geometry_ufno as suite


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run final CFO vs Geometry-U-FNO diagnostics "
            "at an arbitrary resolution."
        )
    )

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
            "Resolution used to train the supplied checkpoints. "
            "If omitted, assumes same as --resolution."
        ),
    )

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

    parser.add_argument(
        "--final-ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

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

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve_path(
    path_string,
):

    path = Path(
        path_string
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# CHECK ID DATA RESOLUTION
# =====================================================================

def verify_id_resolution(
    path,
    resolution,
):

    path = resolve_path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"ID dataset not found:\n{path}"
        )

    with h5py.File(
        path,
        "r",
    ) as h5:

        if (
            "test_id"
            not in h5
        ):

            raise KeyError(
                f"{path} does not contain "
                "'test_id'."
            )

        q_shape = (
            h5[
                "test_id/q"
            ]
            .shape
        )

    nx = int(
        q_shape[
            2
        ]
    )

    ny = int(
        q_shape[
            3
        ]
    )

    if (
        nx
        !=
        resolution
        or
        ny
        !=
        resolution
    ):

        raise ValueError(
            "Resolution mismatch.\n"
            f"Requested: {resolution} x {resolution}\n"
            f"Dataset:   {nx} x {ny}"
        )

    return (
        nx,
        ny,
        q_shape,
    )


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

    (
        nx,
        ny,
        q_shape,
    ) = verify_id_resolution(
        args.id_data,
        resolution,
    )

    # -----------------------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------------------

    x_min = -2.5
    x_max = 2.5

    y_min = -2.5
    y_max = 2.5

    dx = (
        x_max
        -
        x_min
    ) / resolution

    dy = (
        y_max
        -
        y_min
    ) / resolution

    # -----------------------------------------------------------------
    # PATCH THE EXISTING SUITE'S RESOLUTION GLOBALS
    # -----------------------------------------------------------------

    suite.RESOLUTION = resolution

    suite.X_MIN = x_min
    suite.X_MAX = x_max

    suite.Y_MIN = y_min
    suite.Y_MAX = y_max

    suite.DX = dx
    suite.DY = dy

    suite.CELL_AREA = (
        dx
        *
        dy
    )

    suite.GRAVITY = 1.0

    # -----------------------------------------------------------------
    # OUTPUT DIRECTORY
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

    mode = (
        "same-resolution"
        if (
            source_resolution
            ==
            resolution
        )
        else
        "zero-shot cross-resolution"
    )

    print()
    print(
        "=" * 78
    )

    print(
        "FINAL RESOLUTION DIAGNOSTIC SUITE"
    )

    print(
        "=" * 78
    )

    print(
        "Mode:",
        mode,
    )

    print(
        "Checkpoint training resolution:",
        f"{source_resolution} x {source_resolution}",
    )

    print(
        "Evaluation resolution:",
        f"{resolution} x {resolution}",
    )

    print(
        "dx:",
        dx,
    )

    print(
        "dy:",
        dy,
    )

    print(
        "ID q shape:",
        q_shape,
    )

    print(
        "=" * 78
    )

    provenance_path = (
        output_dir
        /
        "resolution_provenance.txt"
    )

    with open(
        provenance_path,
        "w",
    ) as file:

        file.write(
            "Final resolution diagnostic suite\n"
        )

        file.write(
            f"mode = {mode}\n"
        )

        file.write(
            "source_resolution = "
            f"{source_resolution}\n"
        )

        file.write(
            "target_resolution = "
            f"{resolution}\n"
        )

        file.write(
            f"dx = {dx}\n"
        )

        file.write(
            f"dy = {dy}\n"
        )

        file.write(
            "lambda_pde = "
            f"{args.lambda_pde}\n"
        )

        file.write(
            "lambda_bed = "
            f"{args.lambda_bed}\n"
        )

        file.write(
            "lambda_wb = "
            f"{args.lambda_wb}\n"
        )

        file.write(
            "tangent_loss = 0\n"
        )

    # -----------------------------------------------------------------
    # FORWARD ARGUMENTS TO EXISTING FINAL SUITE
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

        "--sample-index",
        str(
            args.sample_index
        ),

        "--direct-case",
        args.direct_case,

        "--cross-section-case",
        args.cross_section_case,

        "--steps-per-segment",
        str(
            args.steps_per_segment
        ),

        "--output-dir",
        str(
            output_dir
        ),
    ]

    old_argv = list(
        sys.argv
    )

    try:

        sys.argv = forwarded

        suite.main()

    finally:

        sys.argv = old_argv

    print()
    print(
        "=" * 78
    )

    print(
        "RESOLUTION SUITE COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "Results:",
        output_dir,
    )

    print(
        "Provenance:",
        provenance_path,
    )


if __name__ == "__main__":

    main()