"""
FINAL ZERO-SHOT SUPPLEMENTARY FOUR FIGURES
==========================================

Adds four final figures to the existing 8-figure zero-shot study.

Only the FINAL methods are shown:

    1. PyClaw 64x64
       - interpolated to 128 when computing numerical errors

    2. Bathy-CFO
       - trained at 64x64
       - evaluated zero-shot at 128x128

    3. Final Canonical Geometry-U-FNO
       - trained at 64x64
       - evaluated zero-shot at 128x128
       - FNO branch directly at 128
       - U-Net branch:
             128 -> 64 using area averaging
             U-Net at 64
             64 -> 128 using cubic reconstruction
       - no native 128 U-Net blend
       - no spectral low-pass filter

    4. PyClaw 128x128 reference

Outputs
-------
09_gaussian_terrain_effect_error.png
10_ood_two_hill_terrain_effect_error.png
11_zero_shot_cross_sections_2d.png
12_zero_shot_free_surface_2d.png

Also:
final_zero_shot_extra_metrics.csv
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
# REUSE FINAL ZERO-SHOT INFRASTRUCTURE
# =====================================================================

import experiments.run_final_zero_shot_8plots as base


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Four additional final zero-shot figures."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # BATHY-CFO
    # -----------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "final_resolution_study/"
            "bathy_cfo/"
            "res64/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    # -----------------------------------------------------------------
    # FINAL GEOMETRY-U-FNO
    # -----------------------------------------------------------------

    parser.add_argument(
        "--final-ckpt",
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

    # -----------------------------------------------------------------
    # FINAL LOSS CONFIGURATION
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
    # EVALUATION
    # -----------------------------------------------------------------

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
        "--gaussian-case",
        type=str,
        default="hill_center",
    )

    parser.add_argument(
        "--ood-case",
        type=str,
        default="two_hills",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "final_project/"
            "zero_shot_64_to_128"
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
# BUILD + RESTORE FINAL MODELS
# =====================================================================

def build_models(
    args,
):

    print()

    print(
        "=" * 80
    )

    print(
        "RESTORING FINAL ZERO-SHOT MODELS"
    )

    print(
        "=" * 80
    )

    # -----------------------------------------------------------------
    # BATHY-CFO
    # -----------------------------------------------------------------

    print()
    print(
        "Restoring Bathy-CFO..."
    )

    cfo_method = (
        base.build_cfo_method()
    )

    cfo_state = (
        base.restore_cfo_state(
            cfo_method,
            args,
        )
    )

    # -----------------------------------------------------------------
    # FINAL CANONICAL GEO-U-FNO
    # -----------------------------------------------------------------

    print()
    print(
        "Restoring final canonical Geo-U-FNO..."
    )

    final_method = (
        base.build_final_method(
            args
        )
    )

    final_state = (
        base.restore_final_state(
            final_method,
            args,
        )
    )

    return (
        cfo_method,
        cfo_state,
        final_method,
        final_state,
    )


# =====================================================================
# LOAD + EVALUATE ONE CASE
# =====================================================================

def evaluate_pair(
    file64,
    file128,
    case_name,
    *,
    cfo_method,
    cfo_state,
    final_method,
    final_state,
    steps_per_segment,
):

    data64 = (
        base.load_pair(
            file64,
            case_name,
        )
    )

    data128 = (
        base.load_pair(
            file128,
            case_name,
        )
    )

    result = (
        base.evaluate_case(
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
                steps_per_segment
            ),
        )
    )

    return result


# =====================================================================
# PYCLAW64 -> 128 INTERPOLATED PAIR
# =====================================================================

def interpolated_pyclaw64_pair(
    result,
):

    terrain64 = (
        result[
            "data64"
        ][
            "terrain"
        ]
    )

    flat64 = (
        result[
            "data64"
        ][
            "flat"
        ]
    )

    terrain128 = (
        result[
            "data128"
        ][
            "terrain"
        ]
    )

    # -----------------------------------------------------------------
    # TERRAIN
    # -----------------------------------------------------------------

    terrain_interp = (
        base.interpolate_state(
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

    # -----------------------------------------------------------------
    # FLAT
    # -----------------------------------------------------------------

    flat_interp = (
        base.interpolate_state(
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

    return (
        terrain_interp,
        flat_interp,
    )


# =====================================================================
# FIGURE 09
# GAUSSIAN TERRAIN-EFFECT ERROR
# =====================================================================

def plot_gaussian_effect_error(
    result,
    *,
    output_path,
):

    terrain128 = (
        result[
            "data128"
        ][
            "terrain"
        ]
    )

    flat128 = (
        result[
            "data128"
        ][
            "flat"
        ]
    )

    time = (
        terrain128[
            "time"
        ]
    )

    # -----------------------------------------------------------------
    # TRUE 128 TERRAIN EFFECT
    # -----------------------------------------------------------------

    true_effect = (
        terrain128[
            "q"
        ]
        -
        flat128[
            "q"
        ]
    )

    # -----------------------------------------------------------------
    # PYCLAW64 INTERPOLATED
    # -----------------------------------------------------------------

    (
        pyclaw64_terrain,
        pyclaw64_flat,
    ) = interpolated_pyclaw64_pair(
        result
    )

    pyclaw64_effect = (
        pyclaw64_terrain
        -
        pyclaw64_flat
    )

    # -----------------------------------------------------------------
    # BATHY-CFO
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # FINAL CANONICAL
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            11,
            6.5,
        )
    )

    # PyClaw128 reference = zero error

    ax.plot(
        time,

        np.zeros_like(
            time
        ),

        linestyle="--",

        linewidth=2,

        label=(
            "PyClaw 128×128 reference"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            pyclaw64_effect,
        ),

        linewidth=2,

        label=(
            "PyClaw 64→128 interpolation"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            cfo_effect,
        ),

        linewidth=2,

        label=(
            "Bathy-CFO 64→128"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            final_effect,
        ),

        linewidth=2,

        label=(
            "Canonical Geo-U-FNO "
            "64→128 (Area↓ + Cubic↑)"
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
        output_path,

        dpi=250,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 10
# OOD TERRAIN-EFFECT ERROR
# =====================================================================

def plot_ood_effect_error(
    result,
    *,
    output_path,
):

    terrain128 = (
        result[
            "data128"
        ][
            "terrain"
        ]
    )

    flat128 = (
        result[
            "data128"
        ][
            "flat"
        ]
    )

    time = (
        terrain128[
            "time"
        ]
    )

    # -----------------------------------------------------------------
    # TRUE EFFECT
    # -----------------------------------------------------------------

    true_effect = (
        terrain128[
            "q"
        ]
        -
        flat128[
            "q"
        ]
    )

    # -----------------------------------------------------------------
    # PYCLAW64
    # -----------------------------------------------------------------

    (
        pyclaw64_terrain,
        pyclaw64_flat,
    ) = interpolated_pyclaw64_pair(
        result
    )

    pyclaw64_effect = (
        pyclaw64_terrain
        -
        pyclaw64_flat
    )

    # -----------------------------------------------------------------
    # BATHY-CFO
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # FINAL CANONICAL
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            11,
            6.5,
        )
    )

    ax.plot(
        time,

        np.zeros_like(
            time
        ),

        linestyle="--",

        linewidth=2,

        label=(
            "PyClaw 128×128 reference"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            pyclaw64_effect,
        ),

        linewidth=2,

        label=(
            "PyClaw 64→128 interpolation"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            cfo_effect,
        ),

        linewidth=2,

        label=(
            "Bathy-CFO 64→128"
        ),
    )

    ax.plot(
        time,

        base.relative_error_vs_time(
            true_effect,
            final_effect,
        ),

        linewidth=2,

        label=(
            "Canonical Geo-U-FNO "
            "64→128 (Area↓ + Cubic↑)"
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
        output_path,

        dpi=250,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 11
# FINAL 2D CROSS-SECTIONS
# =====================================================================

def plot_final_cross_sections(
    result,
    *,
    plot_time,
    output_path,
):

    terrain64 = (
        result[
            "data64"
        ][
            "terrain"
        ]
    )

    terrain128 = (
        result[
            "data128"
        ][
            "terrain"
        ]
    )

    # -----------------------------------------------------------------
    # TIME INDICES
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # CENTER INDICES
    # -----------------------------------------------------------------

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

    # =================================================================
    # CHANNEL LOOP
    # =================================================================

    for channel in range(
        3
    ):

        # =============================================================
        # X SECTION
        # =============================================================

        ax = axes[
            0,
            channel
        ]

        # PyClaw64

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

            linewidth=2,

            label=(
                "PyClaw 64×64"
            ),
        )

        # Bathy-CFO

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

            linewidth=2,

            label=(
                "Bathy-CFO 64→128"
            ),
        )

        # Final Canonical

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

            linewidth=2,

            label=(
                "Canonical Geo-U-FNO "
                "64→128"
            ),
        )

        # PyClaw128

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

            linewidth=2,

            label=(
                "PyClaw 128×128"
            ),
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_title(
            (
                channel_names[
                    channel
                ]
                +
                " — x section"
            )
        )

        ax.grid(
            alpha=0.25
        )

        # =============================================================
        # Y SECTION
        # =============================================================

        ax = axes[
            1,
            channel
        ]

        # PyClaw64

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

            linewidth=2,

            label=(
                "PyClaw 64×64"
            ),
        )

        # Bathy-CFO

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

            linewidth=2,

            label=(
                "Bathy-CFO 64→128"
            ),
        )

        # Final Canonical

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

            linewidth=2,

            label=(
                "Canonical Geo-U-FNO "
                "64→128"
            ),
        )

        # PyClaw128

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

            linewidth=2,

            label=(
                "PyClaw 128×128"
            ),
        )

        ax.set_xlabel(
            "y"
        )

        ax.set_title(
            (
                channel_names[
                    channel
                ]
                +
                " — y section"
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
            "Final 64→128 zero-shot comparison "
            f"at t={terrain128['time'][index128]:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=250,

        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# FIGURE 12
# FINAL 2D FREE-SURFACE COMPARISON
# =====================================================================

def plot_final_free_surface(
    result,
    *,
    plot_time,
    output_path,
):

    terrain64 = (
        result[
            "data64"
        ][
            "terrain"
        ]
    )

    terrain128 = (
        result[
            "data128"
        ][
            "terrain"
        ]
    )

    # -----------------------------------------------------------------
    # TIME
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # FREE SURFACE
    # eta = h + b
    # -----------------------------------------------------------------

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

    fields = [
        (
            "PyClaw 64×64",
            eta64,
        ),

        (
            "Bathy-CFO 64→128",
            eta_cfo,
        ),

        (
            (
                "Canonical Geo-U-FNO 64→128\n"
                "Area↓ + Cubic↑"
            ),
            eta_final,
        ),

        (
            "PyClaw 128×128",
            eta128,
        ),
    ]

    # -----------------------------------------------------------------
    # COMMON SCALE
    # -----------------------------------------------------------------

    vmin = min(
        float(
            np.min(
                field
            )
        )

        for _, field
        in fields
    )

    vmax = max(
        float(
            np.max(
                field
            )
        )

        for _, field
        in fields
    )

    # -----------------------------------------------------------------
    # PLOT
    # -----------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            20,
            5,
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
                base.X_MIN,
                base.X_MAX,
                base.Y_MIN,
                base.Y_MAX,
            ],

            vmin=vmin,

            vmax=vmax,

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
            "Final free-surface zero-shot comparison "
            f"at t={terrain128['time'][index128]:.2f}"
        ),
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,

        dpi=250,

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
            base.relative_l2(
                truth,
                prediction,
            ),

        "h_rel_l2":
            base.relative_l2(
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
            base.relative_l2(
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
            base.relative_l2(
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
            base.relative_l2(
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
# SAVE EXTRA METRICS
# =====================================================================

def save_metrics(
    gaussian_result,
    ood_result,
    *,
    output_path,
):

    rows = []

    for (
        case_name,
        result,
    ) in [
        (
            "gaussian_hill_center",
            gaussian_result,
        ),
        (
            "ood_two_hills",
            ood_result,
        ),
    ]:

        terrain128 = (
            result[
                "data128"
            ][
                "terrain"
            ]
        )

        flat128 = (
            result[
                "data128"
            ][
                "flat"
            ]
        )

        # -------------------------------------------------------------
        # PYCLAW64 INTERPOLATED
        # -------------------------------------------------------------

        (
            q64,
            flat64,
        ) = interpolated_pyclaw64_pair(
            result
        )

        rows.append(
            metric_row(
                case_name,

                "PyClaw64 interpolated",

                terrain128[
                    "q"
                ],

                q64,

                flat128[
                    "q"
                ],

                flat64,
            )
        )

        # -------------------------------------------------------------
        # BATHY-CFO
        # -------------------------------------------------------------

        rows.append(
            metric_row(
                case_name,

                "Bathy-CFO 64→128",

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

        # -------------------------------------------------------------
        # FINAL CANONICAL
        # -------------------------------------------------------------

        rows.append(
            metric_row(
                case_name,

                (
                    "Canonical Geo-U-FNO "
                    "64→128 Area+Cubic"
                ),

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

        # -------------------------------------------------------------
        # PYCLAW128
        # -------------------------------------------------------------

        rows.append(
            metric_row(
                case_name,

                "PyClaw128 reference",

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
    # MODELS
    # =================================================================

    (
        cfo_method,
        cfo_state,
        final_method,
        final_state,
    ) = build_models(
        args
    )

    # =================================================================
    # GAUSSIAN CENTER CASE
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL GAUSSIAN ZERO-SHOT CASE"
    )

    print(
        "=" * 80
    )

    gaussian_result = (
        evaluate_pair(
            args.gaussian64,
            args.gaussian128,
            args.gaussian_case,

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
    )

    # =================================================================
    # OOD TWO-HILL CASE
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL OOD ZERO-SHOT CASE"
    )

    print(
        "=" * 80
    )

    ood_result = (
        evaluate_pair(
            args.ood64,
            args.ood128,
            args.ood_case,

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
    )

    # =================================================================
    # FIGURE 09
    # =================================================================

    print()
    print(
        "Creating Figure 09..."
    )

    plot_gaussian_effect_error(
        gaussian_result,

        output_path=(
            output_dir
            /
            "09_gaussian_terrain_effect_error.png"
        ),
    )

    # =================================================================
    # FIGURE 10
    # =================================================================

    print(
        "Creating Figure 10..."
    )

    plot_ood_effect_error(
        ood_result,

        output_path=(
            output_dir
            /
            "10_ood_two_hill_terrain_effect_error.png"
        ),
    )

    # =================================================================
    # FIGURE 11
    # =================================================================

    print(
        "Creating Figure 11..."
    )

    plot_final_cross_sections(
        gaussian_result,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "11_zero_shot_cross_sections_2d.png"
        ),
    )

    # =================================================================
    # FIGURE 12
    # =================================================================

    print(
        "Creating Figure 12..."
    )

    plot_final_free_surface(
        gaussian_result,

        plot_time=(
            args.plot_time
        ),

        output_path=(
            output_dir
            /
            "12_zero_shot_free_surface_2d.png"
        ),
    )

    # =================================================================
    # METRICS
    # =================================================================

    save_metrics(
        gaussian_result,
        ood_result,

        output_path=(
            output_dir
            /
            "final_zero_shot_extra_metrics.csv"
        ),
    )

    # =================================================================
    # COMPLETE
    # =================================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL EXTRA ZERO-SHOT FIGURES COMPLETE"
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