"""
Tangent / geometry diagnostic plots.

This script is retained primarily for diagnostic and architecture-selection
analysis. Tangent regularization does not need to remain part of the final
model methodology.

Models
------
1. Geometry-U-FNO baseline
2. Tangent Geometry-U-FNO lambda = 0.001
3. Tangent Geometry-U-FNO lambda = 0.005
4. Tangent Geometry-U-FNO lambda = 0.01

Outputs
-------
01_direct_condition_test.png
02_isolated_hill_effect_error.png
02_isolated_hill_effect_error.csv
03_terrain_induced_speed_gaussian.png
04_terrain_induced_speed_unseen.png
05_gaussian_cross_sections_h_hu_hv.png
06_gaussian_3d_bathymetry_cross_sections.png
07_gaussian_3d_free_surface.png
08_gaussian_3d_h_hu_hv_cross_sections.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# =====================================================================
# PROJECT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.geometry_ufno import GeometryUFNO2d

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import init_cfo_train_state

from utils.checkpoints import load_train_state

import experiments.plot_wb_bed_diagnostics as diag


# =====================================================================
# GRID / PHYSICS
# =====================================================================

RESOLUTION = 32

X_MIN = -2.5
X_MAX = 2.5

Y_MIN = -2.5
Y_MAX = 2.5

DX = (X_MAX - X_MIN) / RESOLUTION
DY = (Y_MAX - Y_MIN) / RESOLUTION

GRAVITY = 1.0


# =====================================================================
# MODEL DEFINITIONS
# =====================================================================

MODEL_KEYS = [
    "baseline",
    "tan001",
    "tan005",
    "tan01",
]


MODEL_LABELS = {

    "baseline":
        "Geometry-U-FNO",

    "tan001":
        r"Tangent U-FNO $\lambda=0.001$",

    "tan005":
        r"Tangent U-FNO $\lambda=0.005$",

    "tan01":
        r"Tangent U-FNO $\lambda=0.01$",
}


CHECKPOINT_DEFAULTS = {

    "baseline":
        (
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "lamwb_0p1/"
            "seed0"
        ),

    "tan001":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p001/"
            "seed0"
        ),

    "tan005":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p005/"
            "seed0"
        ),

    "tan01":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p01/"
            "seed0"
        ),
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--counterfactual-data",
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_32.h5"
        ),
    )

    parser.add_argument(
        "--ood-data",
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_32.h5"
        ),
    )

    parser.add_argument(
        "--baseline-checkpoint",
        default=CHECKPOINT_DEFAULTS[
            "baseline"
        ],
    )

    parser.add_argument(
        "--tan001-checkpoint",
        default=CHECKPOINT_DEFAULTS[
            "tan001"
        ],
    )

    parser.add_argument(
        "--tan005-checkpoint",
        default=CHECKPOINT_DEFAULTS[
            "tan005"
        ],
    )

    parser.add_argument(
        "--tan01-checkpoint",
        default=CHECKPOINT_DEFAULTS[
            "tan01"
        ],
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "tangent_geometry_diagnostics"
        ),
    )

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
        default="hill_right",
        choices=[
            "hill_left",
            "hill_center",
            "hill_right",
        ],
    )

    parser.add_argument(
        "--cross-section-case",
        default="hill_center",
        choices=[
            "hill_left",
            "hill_center",
            "hill_right",
        ],
    )

    return parser.parse_args()


# =====================================================================
# HELPERS
# =====================================================================

def resolve_path(value):

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def configure_diag_globals():

    diag.NX = RESOLUTION
    diag.NY = RESOLUTION

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = X_MIN
    diag.X_MAX = X_MAX

    diag.Y_MIN = Y_MIN
    diag.Y_MAX = Y_MAX


def cell_coordinates():

    x = (
        X_MIN
        +
        (
            np.arange(RESOLUTION)
            +
            0.5
        )
        *
        DX
    )

    y = (
        Y_MIN
        +
        (
            np.arange(RESOLUTION)
            +
            0.5
        )
        *
        DY
    )

    return x, y


def hill_center_indices(b):

    index = np.unravel_index(
        np.argmax(b),
        b.shape,
    )

    return (
        int(index[0]),
        int(index[1]),
    )


def bathymetry_contour_legend():

    return Line2D(
        [0],
        [0],
        linewidth=2,
        label="Bathymetry contours",
    )


# =====================================================================
# MODEL
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


def restore_checkpoint(
    method,
    root,
):

    root = resolve_path(
        root
    )

    if not root.exists():

        raise FileNotFoundError(
            f"Checkpoint does not exist:\n"
            f"{root}"
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
        ckpt_dir=str(root),
        prefix="best",
        step=None,
        max_to_keep=1,
    )


def build_all_models(args):

    checkpoint_paths = {

        "baseline":
            args.baseline_checkpoint,

        "tan001":
            args.tan001_checkpoint,

        "tan005":
            args.tan005_checkpoint,

        "tan01":
            args.tan01_checkpoint,
    }

    methods = {}
    states = {}

    for key in MODEL_KEYS:

        print()
        print(
            "Restoring",
            MODEL_LABELS[key],
        )

        method = build_method()

        state = restore_checkpoint(
            method,
            checkpoint_paths[key],
        )

        methods[key] = method
        states[key] = state

    return methods, states


# =====================================================================
# CASE EVALUATION
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

    models = {}

    for key in MODEL_KEYS:

        print(
            f"Running "
            f"{MODEL_LABELS[key]} "
            f"on {case_name}"
        )

        (
            hill_prediction,
            flat_prediction,
        ) = diag.model_pair_rollout(
            methods[key],
            states[key],
            data,
            steps_per_segment=(
                steps_per_segment
            ),
        )

        models[key] = {
            "hill":
                hill_prediction,

            "flat":
                flat_prediction,
        }

    return {
        "data":
            data,

        "models":
            models,
    }


# =====================================================================
# PLOT 01
# DIRECT CONDITION RESPONSE
# =====================================================================

def plot_direct_condition(
    result,
    *,
    methods,
    states,
    plot_time,
    output_path,
):

    data = result["data"]

    time = data["time"]

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
        time[index]
    )

    q = data[
        "terrain_q"
    ][index]

    b = data["b"]

    b_zero = np.zeros_like(b)

    h = np.maximum(
        q[..., 0],
        1.0e-6,
    )

    expected_hu = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_x(b)
    )

    expected_hv = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_y(b)
    )

    responses = {}

    for key in MODEL_KEYS:

        with_bed = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q,
                b,
                time_value,
            )
        )

        without_bed = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q,
                b_zero,
                time_value,
            )
        )

        responses[key] = (
            with_bed
            -
            without_bed
        )

    hu_fields = [
        expected_hu,
        *[
            responses[key][..., 1]
            for key in MODEL_KEYS
        ],
    ]

    hv_fields = [
        expected_hv,
        *[
            responses[key][..., 2]
            for key in MODEL_KEYS
        ],
    ]

    labels = [
        "SWE expected",
        *[
            MODEL_LABELS[key]
            for key in MODEL_KEYS
        ],
    ]

    hu_limit = max(
        max(
            float(
                np.max(
                    np.abs(field)
                )
            )
            for field in hu_fields
        ),
        1.0e-8,
    )

    hv_limit = max(
        max(
            float(
                np.max(
                    np.abs(field)
                )
            )
            for field in hv_fields
        ),
        1.0e-8,
    )

    n_columns = len(labels)

    fig, axes = plt.subplots(
        2,
        n_columns,
        figsize=(
            4.5 * n_columns,
            9,
        ),
    )

    for column in range(
        n_columns
    ):

        image = diag.heatmap(
            axes[
                0,
                column
            ],
            hu_fields[
                column
            ],
            vlim=hu_limit,
            title=(
                labels[column]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),
            b=b,
        )

        colorbar = fig.colorbar(
            image,
            ax=axes[
                0,
                column
            ],
            shrink=0.80,
        )

        colorbar.set_label(
            r"$\Delta(hu)_t$"
        )

        image = diag.heatmap(
            axes[
                1,
                column
            ],
            hv_fields[
                column
            ],
            vlim=hv_limit,
            title=(
                labels[column]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),
            b=b,
        )

        colorbar = fig.colorbar(
            image,
            ax=axes[
                1,
                column
            ],
            shrink=0.80,
        )

        colorbar.set_label(
            r"$\Delta(hv)_t$"
        )

    fig.legend(
        handles=[
            bathymetry_contour_legend()
        ],
        loc="upper right",
        frameon=True,
    )

    fig.suptitle(
        (
            "Direct condition response\n"
            "same q,t; change only b; "
            f"t={time_value:.2f}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.92,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# PLOT 02
# ISOLATED HILL ERROR
# =====================================================================

def plot_isolated_hill_errors(
    results,
    *,
    output_path,
    csv_path,
):

    cases = [
        "hill_left",
        "hill_center",
        "hill_right",
    ]

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
        cases
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

        ax = axes[
            column
        ]

        for key in MODEL_KEYS:

            model = result[
                "models"
            ][key]

            error = (
                diag.relative_effect_error(
                    data["terrain_q"],
                    data["flat_q"],
                    model["hill"],
                    model["flat"],
                )
            )

            ax.plot(
                time,
                error,
                linewidth=2,
                label=MODEL_LABELS[key],
            )

            for index in range(
                len(time)
            ):

                rows.append(
                    {
                        "case":
                            case_name,

                        "time":
                            float(
                                time[index]
                            ),

                        "model":
                            MODEL_LABELS[key],

                        "relative_effect_error":
                            float(
                                error[index]
                            ),
                    }
                )

        ax.set_title(case_name)

        ax.set_xlabel("Time")

        ax.set_ylabel(
            (
                "Relative error in "
                "isolated terrain effect"
            )
        )

        ax.grid(alpha=0.30)

        ax.legend(
            fontsize=8,
            loc="best",
            frameon=True,
        )

    fig.suptitle(
        (
            "Error in the isolated "
            "Gaussian-hill effect"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

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

        writer.writerows(rows)


# =====================================================================
# PLOTS 03 + 04
# TERRAIN-INDUCED SPEED
# =====================================================================

def plot_speed_grid(
    results,
    *,
    case_names,
    plot_time,
    output_path,
    title,
):

    n_rows = len(case_names)

    n_columns = (
        1
        +
        len(MODEL_KEYS)
    )

    labels = [
        "PyClaw",
        *[
            MODEL_LABELS[key]
            for key in MODEL_KEYS
        ],
    ]

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.3 * n_columns,
            4.2 * n_rows,
        ),
    )

    if n_rows == 1:
        axes = np.asarray(
            axes
        )[
            None,
            :
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
                data["terrain_q"],
                data["flat_q"],
                index,
            )
        ]

        for key in MODEL_KEYS:

            model = result[
                "models"
            ][key]

            fields.append(
                diag.terrain_speed_effect(
                    model["hill"],
                    model["flat"],
                    index,
                )
            )

        value_limit = max(
            max(
                float(
                    np.max(
                        np.abs(field)
                    )
                )
                for field in fields
            ),
            1.0e-8,
        )

        for column in range(
            n_columns
        ):

            image = diag.heatmap(
                axes[
                    row,
                    column
                ],
                fields[
                    column
                ],
                vlim=value_limit,
                title=(
                    case_name
                    +
                    "\n"
                    +
                    labels[column]
                ),
                b=data["b"],
            )

            colorbar = fig.colorbar(
                image,
                ax=axes[
                    row,
                    column
                ],
                shrink=0.76,
            )

            colorbar.set_label(
                (
                    r"$|\mathbf{u}|_{terrain}"
                    r"-|\mathbf{u}|_{flat}$"
                )
            )

    fig.legend(
        handles=[
            bathymetry_contour_legend()
        ],
        loc="upper right",
        frameon=True,
    )

    fig.suptitle(
        (
            title
            +
            f" at t = {plot_time:.2f}"
        ),
        fontsize=18,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# PLOT 05
# CROSS-SECTIONS
#
# NOTE:
# NO bathymetry Gaussian/dashed profile is shown here anymore.
# =====================================================================

def plot_cross_sections(
    result,
    *,
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

    truth = data[
        "terrain_q"
    ][
        time_index
    ]

    b = data[
        "b"
    ]

    ix, iy = hill_center_indices(
        b
    )

    x, y = cell_coordinates()

    variables = [
        ("h", 0),
        ("hu", 1),
        ("hv", 2),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            9,
        ),
    )

    for column, (
        variable_name,
        variable_index,
    ) in enumerate(
        variables
    ):

        # =========================================================
        # y = constant
        # =========================================================

        ax = axes[
            0,
            column
        ]

        ax.plot(
            x,
            truth[
                :,
                iy,
                variable_index
            ],
            linewidth=2.5,
            label="PyClaw",
        )

        for key in MODEL_KEYS:

            prediction = (
                result[
                    "models"
                ][key][
                    "hill"
                ][
                    time_index
                ]
            )

            ax.plot(
                x,
                prediction[
                    :,
                    iy,
                    variable_index
                ],
                linewidth=1.7,
                label=MODEL_LABELS[key],
            )

        ax.set_xlabel("x")

        ax.set_ylabel(
            variable_name
        )

        ax.set_title(
            (
                f"{variable_name} along "
                f"y={y[iy]:.3f}"
            )
        )

        ax.grid(alpha=0.3)

        ax.legend(
            fontsize=7,
            loc="best",
            frameon=True,
        )

        # =========================================================
        # x = constant
        # =========================================================

        ax = axes[
            1,
            column
        ]

        ax.plot(
            y,
            truth[
                ix,
                :,
                variable_index
            ],
            linewidth=2.5,
            label="PyClaw",
        )

        for key in MODEL_KEYS:

            prediction = (
                result[
                    "models"
                ][key][
                    "hill"
                ][
                    time_index
                ]
            )

            ax.plot(
                y,
                prediction[
                    ix,
                    :,
                    variable_index
                ],
                linewidth=1.7,
                label=MODEL_LABELS[key],
            )

        ax.set_xlabel("y")

        ax.set_ylabel(
            variable_name
        )

        ax.set_title(
            (
                f"{variable_name} along "
                f"x={x[ix]:.3f}"
            )
        )

        ax.grid(alpha=0.3)

        ax.legend(
            fontsize=7,
            loc="best",
            frameon=True,
        )

    fig.suptitle(
        (
            "Cross-sections through Gaussian hill "
            f"at t={time_value:.2f}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# PLOT 06
# 3D BATHYMETRY
# =====================================================================

def plot_3d_bathymetry(
    result,
    *,
    output_path,
):

    data = result[
        "data"
    ]

    b = data[
        "b"
    ]

    x, y = cell_coordinates()

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    ix, iy = hill_center_indices(
        b
    )

    fig = plt.figure(
        figsize=(
            9,
            7,
        )
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.plot_surface(
        X,
        Y,
        b,
        color="steelblue",
        alpha=0.70,
        linewidth=0,
        antialiased=True,
    )

    line_y, = ax.plot(
        x,
        np.full_like(
            x,
            y[iy],
        ),
        b[
            :,
            iy
        ],
        linewidth=3,
        label=(
            f"y={y[iy]:.3f} section"
        ),
    )

    line_x, = ax.plot(
        np.full_like(
            y,
            x[ix],
        ),
        y,
        b[
            ix,
            :
        ],
        linewidth=3,
        linestyle="--",
        label=(
            f"x={x[ix]:.3f} section"
        ),
    )

    surface_proxy = Patch(
        facecolor="steelblue",
        alpha=0.70,
        label="Gaussian bathymetry",
    )

    ax.legend(
        handles=[
            surface_proxy,
            line_y,
            line_x,
        ],
        loc="upper right",
        frameon=True,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("b")

    ax.set_title(
        (
            "3D Gaussian bathymetry "
            "with cross-section paths"
        )
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# PLOT 07
# 3D FREE SURFACE
# =====================================================================

def plot_3d_free_surface(
    result,
    *,
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

    b = data[
        "b"
    ]

    x, y = cell_coordinates()

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    surfaces = [

        (
            "PyClaw",

            data[
                "terrain_q"
            ][
                time_index,
                ...,
                0
            ]
            +
            b,
        )
    ]

    for key in MODEL_KEYS:

        q = (
            result[
                "models"
            ][key][
                "hill"
            ][
                time_index
            ]
        )

        surfaces.append(
            (
                MODEL_LABELS[key],

                q[
                    ...,
                    0
                ]
                +
                b,
            )
        )

    n_columns = len(
        surfaces
    )

    # -------------------------------------------------------------
    # Explicit matching colors.
    # Brown = free surface.
    # Blue  = bathymetry.
    # -------------------------------------------------------------

    free_surface_color = (
        "peru"
    )

    bathymetry_color = (
        "lightsteelblue"
    )

    fig = plt.figure(
        figsize=(
            5
            *
            n_columns,
            5,
        )
    )

    for column, (
        label,
        eta,
    ) in enumerate(
        surfaces,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            n_columns,
            column,
            projection="3d",
        )

        # =========================================================
        # BLUE BATHYMETRY
        # =========================================================

        ax.plot_surface(
            X,
            Y,
            b,
            color=bathymetry_color,
            alpha=0.50,
            linewidth=0,
        )

        # =========================================================
        # BROWN FREE SURFACE
        # =========================================================

        ax.plot_surface(
            X,
            Y,
            eta,
            color=free_surface_color,
            alpha=0.82,
            linewidth=0,
        )

        # =========================================================
        # LEGEND WITH MATCHING COLORS
        # =========================================================

        free_surface_proxy = Patch(
            facecolor=free_surface_color,
            edgecolor="black",
            alpha=0.82,
            label=(
                r"Free surface "
                r"$\eta=h+b$"
            ),
        )

        bathymetry_proxy = Patch(
            facecolor=bathymetry_color,
            edgecolor="black",
            alpha=0.50,
            label=(
                r"Bathymetry $b$"
            ),
        )

        ax.legend(
            handles=[
                free_surface_proxy,
                bathymetry_proxy,
            ],
            loc="upper right",
            fontsize=7,
            frameon=True,
        )

        ax.set_title(label)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_zlabel(
            r"$\eta$"
        )

    fig.suptitle(
        (
            "3D free surface over Gaussian hill "
            f"at t={time[time_index]:.2f}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# PLOT 08
# 3D VERSION OF h / hu / hv CROSS-SECTIONS
# =====================================================================

def plot_3d_state_cross_sections(
    result,
    *,
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

    b = data[
        "b"
    ]

    ix, iy = hill_center_indices(
        b
    )

    x, y = cell_coordinates()

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    model_fields = {

        "PyClaw":
            data[
                "terrain_q"
            ][
                time_index
            ],
    }

    for key in MODEL_KEYS:

        model_fields[
            MODEL_LABELS[key]
        ] = (
            result[
                "models"
            ][key][
                "hill"
            ][
                time_index
            ]
        )

    column_labels = list(
        model_fields.keys()
    )

    variables = [
        ("h", 0),
        ("hu", 1),
        ("hv", 2),
    ]

    n_rows = len(
        variables
    )

    n_columns = len(
        column_labels
    )

    z_limits = {}

    for variable_name, channel in variables:

        all_values = np.concatenate(
            [
                model_fields[
                    label
                ][
                    ...,
                    channel
                ].ravel()

                for label
                in column_labels
            ]
        )

        value_min = float(
            np.min(
                all_values
            )
        )

        value_max = float(
            np.max(
                all_values
            )
        )

        span = max(
            value_max
            -
            value_min,
            1.0e-6,
        )

        margin = (
            0.08
            *
            span
        )

        z_limits[
            variable_name
        ] = (
            value_min
            -
            margin,

            value_max
            +
            margin,
        )

    fig = plt.figure(
        figsize=(
            5.0
            *
            n_columns,

            4.6
            *
            n_rows,
        )
    )

    for row, (
        variable_name,
        channel,
    ) in enumerate(
        variables
    ):

        z_min, z_max = (
            z_limits[
                variable_name
            ]
        )

        z_span = max(
            z_max
            -
            z_min,
            1.0e-6,
        )

        projection_floor = (
            z_min
            -
            0.06
            *
            z_span
        )

        path_lift = (
            0.015
            *
            z_span
        )

        for column, label in enumerate(
            column_labels
        ):

            field = (
                model_fields[
                    label
                ][
                    ...,
                    channel
                ]
            )

            subplot_index = (
                row
                *
                n_columns
                +
                column
                +
                1
            )

            ax = fig.add_subplot(
                n_rows,
                n_columns,
                subplot_index,
                projection="3d",
            )

            # =====================================================
            # FULL STATE FIELD
            # =====================================================

            ax.plot_surface(
                X,
                Y,
                field,
                color="steelblue",
                linewidth=0,
                antialiased=True,
                alpha=0.62,
            )

            # =====================================================
            # y = constant section
            # =====================================================

            section_y, = ax.plot(
                x,

                np.full_like(
                    x,
                    y[iy],
                ),

                field[
                    :,
                    iy
                ]
                +
                path_lift,

                linewidth=3,

                label=(
                    f"y={y[iy]:.3f} section"
                ),
            )

            # =====================================================
            # x = constant section
            # =====================================================

            section_x, = ax.plot(
                np.full_like(
                    y,
                    x[ix],
                ),

                y,

                field[
                    ix,
                    :
                ]
                +
                path_lift,

                linewidth=3,

                linestyle="--",

                label=(
                    f"x={x[ix]:.3f} section"
                ),
            )

            # =====================================================
            # BATHYMETRY CONTOURS
            # =====================================================

            b_max = float(
                np.max(b)
            )

            if b_max > 1.0e-12:

                positive_b = b[
                    b
                    >
                    0.0
                ]

                if positive_b.size > 0:

                    contour_levels = np.linspace(
                        max(
                            float(
                                np.min(
                                    positive_b
                                )
                            ),
                            0.10
                            *
                            b_max,
                        ),
                        b_max,
                        6,
                    )

                    ax.contour(
                        X,
                        Y,
                        b,
                        levels=contour_levels,
                        zdir="z",
                        offset=projection_floor,
                        linewidths=1.0,
                    )

            surface_proxy = Patch(
                facecolor="steelblue",
                alpha=0.62,
                label=(
                    f"{variable_name}(x,y)"
                ),
            )

            bathymetry_proxy = (
                Line2D(
                    [0],
                    [0],
                    linewidth=1.8,
                    linestyle=":",
                    label=(
                        "Bathymetry contours"
                    ),
                )
            )

            ax.legend(
                handles=[
                    surface_proxy,
                    section_y,
                    section_x,
                    bathymetry_proxy,
                ],
                loc="upper right",
                fontsize=6.3,
                frameon=True,
            )

            if row == 0:

                ax.set_title(
                    label,
                    fontsize=10,
                    pad=12,
                )

            ax.set_xlabel(
                "x",
                fontsize=8,
            )

            ax.set_ylabel(
                "y",
                fontsize=8,
            )

            ax.set_zlabel(
                variable_name,
                fontsize=8,
            )

            ax.set_zlim(
                projection_floor,
                z_max,
            )

            ax.tick_params(
                labelsize=7
            )

            ax.view_init(
                elev=28,
                azim=-58,
            )

    fig.suptitle(
        (
            "3D h, hu, hv fields with Gaussian-hill "
            "cross-section paths\n"
            f"t={time_value:.2f}, "
            f"y-section={y[iy]:.3f}, "
            f"x-section={x[ix]:.3f}"
        ),
        fontsize=18,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    fig.savefig(
        output_path,
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    configure_diag_globals()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods, states = (
        build_all_models(
            args
        )
    )

    gaussian_archive = (
        diag.H5Archive(
            args.counterfactual_data
        )
    )

    ood_archive = (
        diag.H5Archive(
            args.ood_data
        )
    )

    try:

        # =============================================================
        # GAUSSIAN
        # =============================================================

        gaussian_results = {}

        for case_name in [
            "hill_left",
            "hill_center",
            "hill_right",
        ]:

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
        # UNSEEN TERRAIN
        # =============================================================

        ood_results = {}

        for case_name in [
            "two_hills",
            "narrow_tall",
            "elongated_ridge",
            "rotated_ridge",
            "multi_hill",
        ]:

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
        # PLOT 01
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
        # PLOT 02
        # =============================================================

        plot_isolated_hill_errors(
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
        # PLOT 03
        # =============================================================

        plot_speed_grid(
            gaussian_results,

            case_names=[
                "hill_left",
                "hill_center",
                "hill_right",
            ],

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
        # PLOT 04
        # =============================================================

        plot_speed_grid(
            ood_results,

            case_names=[
                "two_hills",
                "narrow_tall",
                "elongated_ridge",
                "rotated_ridge",
                "multi_hill",
            ],

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
        # COMMON GAUSSIAN CASE
        # =============================================================

        cross_result = (
            gaussian_results[
                args.cross_section_case
            ]
        )

        # =============================================================
        # PLOT 05
        # =============================================================

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

        # =============================================================
        # PLOT 06
        # =============================================================

        plot_3d_bathymetry(
            cross_result,

            output_path=(
                output_dir
                /
                "06_gaussian_3d_bathymetry_cross_sections.png"
            ),
        )

        # =============================================================
        # PLOT 07
        # =============================================================

        plot_3d_free_surface(
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

        # =============================================================
        # PLOT 08
        # =============================================================

        plot_3d_state_cross_sections(
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

    finally:

        gaussian_archive.close()
        ood_archive.close()

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
        "Results:",
        output_dir
    )


if __name__ == "__main__":
    main()