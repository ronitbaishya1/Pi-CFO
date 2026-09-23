"""
FINAL MODEL COMPARISON
======================

Compare:

    1. PyClaw reference
    2. Bathymetry-conditioned CFO
    3. Final Geometry-U-FNO + WB-Bed-PI-CFO

The final Geometry-U-FNO uses

    L_total
    =
    L_CFO
    + 0.03 * L_PDE
    + 0.70 * L_bed
    + 0.10 * L_WB

and GeometryEncoder2d features

    [b, b_x, b_y]

No tangent loss is used.

Outputs
-------

TABLES
------
table_00_model_configuration.csv/.md
table_01_id_metrics.csv/.md
table_02_gaussian_rollout_metrics.csv/.md
table_03_gaussian_mass_drift.csv/.md
table_04_gaussian_terrain_effect.csv/.md
table_05_direct_condition_metrics.csv/.md
table_06_ood_rollout_metrics.csv/.md
table_07_ood_terrain_effect.csv/.md
table_08_final_summary.csv/.md


FIGURES
-------
01_direct_condition_test.png
02_isolated_hill_effect_error.png
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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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

from cfo import (
    ContinuousFlowOperator,
)

from models.factory import (
    build_model,
)

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

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

import experiments.plot_wb_bed_diagnostics as diag


# =====================================================================
# CONSTANTS
# =====================================================================

RESOLUTION = 32

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

CELL_AREA = (
    DX
    *
    DY
)

GRAVITY = 1.0


GAUSSIAN_CASES = [
    "hill_left",
    "hill_center",
    "hill_right",
]


OOD_CASES = [
    "two_hills",
    "narrow_tall",
    "elongated_ridge",
    "rotated_ridge",
    "multi_hill",
]


MODEL_KEYS = [
    "cfo",
    "final",
]


MODEL_LABELS = {

    "cfo":
        "Bathy-CFO",

    "final":
        "Final Geometry-U-FNO",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Final PyClaw vs CFO vs "
            "Geometry-U-FNO diagnostics."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    parser.add_argument(
        "--id-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
    )

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
    # CFO CHECKPOINT
    #
    # Training used:
    #
    # checkpoints/bathy_cfo_full/seed0/best
    # prefix = bathy_cfo
    # -----------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_cfo_full/"
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
    #
    # Its checkpoint manager is:
    #
    # .../seed0/best
    #
    # where "best" is the checkpoint prefix.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--final-ckpt",
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

    # -----------------------------------------------------------------
    # FINAL LOSS CONFIGURATION
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

    parser.add_argument(
        "--cross-section-case",
        type=str,
        default="hill_center",
        choices=GAUSSIAN_CASES,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "final_cfo_vs_geometry_ufno"
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
# CONFIGURE EXISTING DIAGNOSTIC UTILITIES
# =====================================================================

def configure_diag():

    diag.NX = RESOLUTION
    diag.NY = RESOLUTION

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = X_MIN
    diag.X_MAX = X_MAX

    diag.Y_MIN = Y_MIN
    diag.Y_MAX = Y_MAX


# =====================================================================
# GRID
# =====================================================================

def cell_coordinates():

    x = (
        X_MIN
        +
        (
            np.arange(
                RESOLUTION
            )
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
            np.arange(
                RESOLUTION
            )
            +
            0.5
        )
        *
        DY
    )

    return (
        x,
        y,
    )


# =====================================================================
# HILL CENTER
# =====================================================================

def hill_center_indices(
    b,
):

    index = np.unravel_index(
        np.argmax(
            b
        ),
        b.shape,
    )

    return (
        int(
            index[0]
        ),

        int(
            index[1]
        ),
    )


# =====================================================================
# BUILD CFO
# =====================================================================

def build_cfo_method():

    input_shape = (
        RESOLUTION,
        RESOLUTION,
        3,
    )

    condition_shape = (
        RESOLUTION,
        RESOLUTION,
        1,
    )

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return ContinuousFlowOperator(
        model=model,

        input_shape=input_shape,

        gamma=1.0e-5,

        spline_type="quintic",

        use_condition=True,

        condition_shape=(
            condition_shape
        ),
    )


# =====================================================================
# BUILD FINAL GEOMETRY-U-FNO
# =====================================================================

def build_final_method(
    args,
):

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

    return (
        WellBalancedBathymetryBedPICFO(
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
    )


# =====================================================================
# RESTORE CFO
# =====================================================================

def restore_cfo_state(
    method,
    args,
):

    checkpoint_dir = resolve_path(
        args.cfo_ckpt_dir
    )

    if not checkpoint_dir.exists():

        raise FileNotFoundError(
            "CFO checkpoint directory "
            "does not exist:\n"
            f"{checkpoint_dir}"
        )

    target = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target,

        ckpt_dir=str(
            checkpoint_dir
        ),

        prefix=(
            args.cfo_prefix
        ),

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# RESTORE FINAL MODEL
# =====================================================================

def restore_final_state(
    method,
    args,
):

    checkpoint_path = resolve_path(
        args.final_ckpt
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Final Geometry-U-FNO "
            "checkpoint does not exist:\n"
            f"{checkpoint_path}"
        )

    checkpoint_root = (
        checkpoint_path.parent
    )

    checkpoint_prefix = (
        checkpoint_path.name
    )

    target = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target,

        ckpt_dir=str(
            checkpoint_root
        ),

        prefix=(
            checkpoint_prefix
        ),

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# BUILD / RESTORE BOTH MODELS
# =====================================================================

def build_models(
    args,
):

    print()
    print(
        "=" * 72
    )
    print(
        "RESTORING FINAL COMPARISON MODELS"
    )
    print(
        "=" * 72
    )

    cfo_method = (
        build_cfo_method()
    )

    print()
    print(
        "Restoring Bathy-CFO..."
    )

    cfo_state = (
        restore_cfo_state(
            cfo_method,
            args,
        )
    )

    final_method = (
        build_final_method(
            args
        )
    )

    print()
    print(
        "Restoring final Geometry-U-FNO..."
    )

    final_state = (
        restore_final_state(
            final_method,
            args,
        )
    )

    methods = {

        "cfo":
            cfo_method,

        "final":
            final_method,
    }

    states = {

        "cfo":
            cfo_state,

        "final":
            final_state,
    }

    return (
        methods,
        states,
    )


# =====================================================================
# BASIC METRICS
# =====================================================================

def relative_l2(
    truth,
    prediction,
    eps=1.0e-12,
):

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

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


def rmse(
    truth,
    prediction,
):

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

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


def relative_frobenius(
    truth,
    prediction,
    eps=1.0e-12,
):

    error = (
        np.asarray(
            prediction,
            dtype=np.float64,
        )
        -
        np.asarray(
            truth,
            dtype=np.float64,
        )
    )

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    return float(
        np.linalg.norm(
            error.ravel()
        )
        /
        max(
            np.linalg.norm(
                truth.ravel()
            ),
            eps,
        )
    )


# =====================================================================
# COSINE
# =====================================================================

def cosine_similarity(
    truth,
    prediction,
    eps=1.0e-12,
):

    a = np.asarray(
        truth,
        dtype=np.float64,
    ).ravel()

    b = np.asarray(
        prediction,
        dtype=np.float64,
    ).ravel()

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator < eps:
        return np.nan

    return float(
        np.dot(
            a,
            b,
        )
        /
        denominator
    )


# =====================================================================
# GAIN RATIO
# =====================================================================

def gain_ratio(
    truth,
    prediction,
    eps=1.0e-12,
):

    truth_norm = np.linalg.norm(
        np.asarray(
            truth,
            dtype=np.float64,
        ).ravel()
    )

    prediction_norm = np.linalg.norm(
        np.asarray(
            prediction,
            dtype=np.float64,
        ).ravel()
    )

    return float(
        prediction_norm
        /
        max(
            truth_norm,
            eps,
        )
    )


# =====================================================================
# NORMALIZE TRAJECTORY SHAPES
# =====================================================================

def ensure_batch_q(
    q,
):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    if q.ndim == 4:

        q = q[
            None,
            ...
        ]

    if q.ndim != 5:

        raise ValueError(
            "Expected q shape "
            "(T,H,W,3) or (N,T,H,W,3), "
            f"got {q.shape}."
        )

    return q


def ensure_batch_b(
    b,
):

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    if b.ndim == 2:

        b = b[
            None,
            ...,
            None
        ]

    elif b.ndim == 3:

        if b.shape[-1] == 1:

            b = b[
                None,
                ...
            ]

        else:

            b = b[
                ...,
                None
            ]

    if b.ndim != 4:

        raise ValueError(
            "Expected bathymetry shape "
            "(H,W), (H,W,1), "
            "(N,H,W) or (N,H,W,1), "
            f"got {b.shape}."
        )

    return b


# =====================================================================
# MASS
# =====================================================================

def mass_series(
    q,
):

    q = ensure_batch_q(
        q
    )

    h = q[
        ...,
        0
    ]

    return (
        np.sum(
            h,
            axis=(
                2,
                3,
            ),
        )
        *
        CELL_AREA
    )


# =====================================================================
# ROLLOUT METRICS
# =====================================================================

def rollout_metrics(
    truth,
    prediction,
    bathymetry,
):

    truth = ensure_batch_q(
        truth
    )

    prediction = ensure_batch_q(
        prediction
    )

    bathymetry = ensure_batch_b(
        bathymetry
    )

    if truth.shape != prediction.shape:

        raise ValueError(
            "Truth/prediction mismatch:\n"
            f"{truth.shape}\n"
            f"{prediction.shape}"
        )

    b2 = bathymetry[
        ...,
        0
    ]

    b_time = b2[
        :,
        None,
        ...,
    ]

    # -----------------------------------------------------------------
    # GLOBAL
    # -----------------------------------------------------------------

    global_rel_l2 = relative_l2(
        truth,
        prediction,
    )

    global_rmse = rmse(
        truth,
        prediction,
    )

    global_rel_fro = (
        relative_frobenius(
            truth,
            prediction,
        )
    )

    # -----------------------------------------------------------------
    # TIME ERROR
    # -----------------------------------------------------------------

    time_errors = []

    for time_index in range(
        truth.shape[1]
    ):

        time_errors.append(
            relative_l2(
                truth[
                    :,
                    time_index
                ],
                prediction[
                    :,
                    time_index
                ],
            )
        )

    time_errors = np.asarray(
        time_errors,
        dtype=np.float64,
    )

    # -----------------------------------------------------------------
    # CHANNELS
    # -----------------------------------------------------------------

    h_rel_l2 = relative_l2(
        truth[
            ...,
            0
        ],
        prediction[
            ...,
            0
        ],
    )

    hu_rel_l2 = relative_l2(
        truth[
            ...,
            1
        ],
        prediction[
            ...,
            1
        ],
    )

    hv_rel_l2 = relative_l2(
        truth[
            ...,
            2
        ],
        prediction[
            ...,
            2
        ],
    )

    momentum_rel_l2 = relative_l2(
        truth[
            ...,
            1:3
        ],
        prediction[
            ...,
            1:3
        ],
    )

    eta_truth = (
        truth[
            ...,
            0
        ]
        +
        b_time
    )

    eta_prediction = (
        prediction[
            ...,
            0
        ]
        +
        b_time
    )

    eta_rel_l2 = relative_l2(
        eta_truth,
        eta_prediction,
    )

    # -----------------------------------------------------------------
    # MASS
    # -----------------------------------------------------------------

    true_mass = mass_series(
        truth
    )

    pred_mass = mass_series(
        prediction
    )

    true_initial_mass = (
        true_mass[
            :,
            0:1
        ]
    )

    pred_initial_mass = (
        pred_mass[
            :,
            0:1
        ]
    )

    true_drift = (
        np.abs(
            true_mass
            -
            true_initial_mass
        )
        /
        np.maximum(
            np.abs(
                true_initial_mass
            ),
            1.0e-12,
        )
    )

    pred_drift = (
        np.abs(
            pred_mass
            -
            pred_initial_mass
        )
        /
        np.maximum(
            np.abs(
                pred_initial_mass
            ),
            1.0e-12,
        )
    )

    mass_error_vs_reference = (
        np.abs(
            pred_mass
            -
            true_mass
        )
        /
        np.maximum(
            np.abs(
                true_mass
            ),
            1.0e-12,
        )
    )

    # -----------------------------------------------------------------
    # POSITIVITY
    # -----------------------------------------------------------------

    minimum_depth = float(
        np.min(
            prediction[
                ...,
                0
            ]
        )
    )

    negative_depth_fraction = float(
        np.mean(
            prediction[
                ...,
                0
            ]
            <
            0.0
        )
    )

    return {

        "rel_l2":
            global_rel_l2,

        "rmse":
            global_rmse,

        "rel_fro":
            global_rel_fro,

        "mean_time_rel_l2":
            float(
                np.mean(
                    time_errors
                )
            ),

        "final_time_rel_l2":
            float(
                time_errors[-1]
            ),

        "h_rel_l2":
            h_rel_l2,

        "hu_rel_l2":
            hu_rel_l2,

        "hv_rel_l2":
            hv_rel_l2,

        "momentum_rel_l2":
            momentum_rel_l2,

        "eta_rel_l2":
            eta_rel_l2,

        "mean_mass_drift":
            float(
                np.mean(
                    pred_drift
                )
            ),

        "final_mass_drift":
            float(
                np.mean(
                    pred_drift[
                        :,
                        -1
                    ]
                )
            ),

        "max_mass_drift":
            float(
                np.max(
                    pred_drift
                )
            ),

        "pyclaw_mean_mass_drift":
            float(
                np.mean(
                    true_drift
                )
            ),

        "pyclaw_final_mass_drift":
            float(
                np.mean(
                    true_drift[
                        :,
                        -1
                    ]
                )
            ),

        "mean_mass_error_vs_pyclaw":
            float(
                np.mean(
                    mass_error_vs_reference
                )
            ),

        "final_mass_error_vs_pyclaw":
            float(
                np.mean(
                    mass_error_vs_reference[
                        :,
                        -1
                    ]
                )
            ),

        "max_mass_error_vs_pyclaw":
            float(
                np.max(
                    mass_error_vs_reference
                )
            ),

        "minimum_depth":
            minimum_depth,

        "negative_depth_fraction":
            negative_depth_fraction,
    }


# =====================================================================
# TERRAIN EFFECT METRICS
# =====================================================================

def terrain_effect_metrics(
    true_hill,
    true_flat,
    pred_hill,
    pred_flat,
    *,
    time_index,
):

    true_effect = (
        true_hill
        -
        true_flat
    )

    pred_effect = (
        pred_hill
        -
        pred_flat
    )

    effect_error_time = (
        diag.relative_effect_error(
            true_hill,
            true_flat,
            pred_hill,
            pred_flat,
        )
    )

    true_speed_effect = (
        diag.terrain_speed_effect(
            true_hill,
            true_flat,
            time_index,
        )
    )

    pred_speed_effect = (
        diag.terrain_speed_effect(
            pred_hill,
            pred_flat,
            time_index,
        )
    )

    return {

        "effect_rel_l2":
            relative_l2(
                true_effect,
                pred_effect,
            ),

        "mean_effect_error":
            float(
                np.mean(
                    effect_error_time
                )
            ),

        "final_effect_error":
            float(
                effect_error_time[-1]
            ),

        "speed_effect_rel_l2":
            relative_l2(
                true_speed_effect,
                pred_speed_effect,
            ),

        "speed_effect_cosine":
            cosine_similarity(
                true_speed_effect,
                pred_speed_effect,
            ),

        "speed_effect_gain":
            gain_ratio(
                true_speed_effect,
                pred_speed_effect,
            ),
    }


# =====================================================================
# TABLE HELPERS
# =====================================================================

def format_value(
    value,
):

    if value is None:

        return ""

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        if not np.isfinite(
            value
        ):
            return "nan"

        if (
            abs(
                value
            )
            <
            1.0e-3
            and
            value
            !=
            0.0
        ):
            return (
                f"{value:.6e}"
            )

        return (
            f"{value:.6f}"
        )

    return str(
        value
    )


def write_table(
    rows,
    *,
    columns,
    csv_path,
    markdown_path,
    title,
):

    csv_path = Path(
        csv_path
    )

    markdown_path = Path(
        markdown_path
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

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
                {
                    column:
                        row.get(
                            column,
                            "",
                        )

                    for column
                    in columns
                }
            )

    # -----------------------------------------------------------------
    # MARKDOWN
    # -----------------------------------------------------------------

    with open(
        markdown_path,
        "w",
    ) as file:

        file.write(
            f"# {title}\n\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                columns
            )
            +
            " |\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                [
                    "---"
                    for _
                    in columns
                ]
            )
            +
            " |\n"
        )

        for row in rows:

            file.write(
                "| "
                +
                " | ".join(
                    [
                        format_value(
                            row.get(
                                column,
                                "",
                            )
                        )

                        for column
                        in columns
                    ]
                )
                +
                " |\n"
            )

    # -----------------------------------------------------------------
    # TERMINAL
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 100
    )
    print(
        title
    )
    print(
        "=" * 100
    )

    widths = {}

    for column in columns:

        widths[
            column
        ] = max(
            len(
                column
            ),

            max(
                [
                    len(
                        format_value(
                            row.get(
                                column,
                                "",
                            )
                        )
                    )

                    for row
                    in rows
                ]
                +
                [0]
            ),
        )

    header = (
        " | ".join(
            [
                f"{column:<{widths[column]}}"
                for column
                in columns
            ]
        )
    )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for row in rows:

        print(
            " | ".join(
                [
                    f"{format_value(row.get(column, '')):<{widths[column]}}"

                    for column
                    in columns
                ]
            )
        )


# =====================================================================
# NORMALIZE BATCH ROLLOUT
# =====================================================================

def normalize_batch_prediction(
    prediction,
    *,
    n_trajectories,
    n_times,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    expected_batch_first = (
        n_trajectories,
        n_times,
        RESOLUTION,
        RESOLUTION,
        3,
    )

    expected_time_first = (
        n_times,
        n_trajectories,
        RESOLUTION,
        RESOLUTION,
        3,
    )

    if (
        prediction.shape
        ==
        expected_batch_first
    ):

        return prediction

    if (
        prediction.shape
        ==
        expected_time_first
    ):

        return np.transpose(
            prediction,
            (
                1,
                0,
                2,
                3,
                4,
            ),
        )

    raise ValueError(
        "Unexpected batch rollout shape: "
        f"{prediction.shape}"
    )


# =====================================================================
# ID ROLLOUT
# =====================================================================

def run_id_rollout(
    method,
    state,
    q,
    bathymetry,
    *,
    steps_per_segment,
):

    q = np.asarray(
        q,
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float32,
    )

    n_trajectories = (
        q.shape[0]
    )

    n_times = (
        q.shape[1]
    )

    prediction = (
        method.uniform_inference(
            state,

            jnp.asarray(
                q[
                    :,
                    0
                ],
                dtype=jnp.float32,
            ),

            trajectory_points_num=(
                n_times
            ),

            steps_per_segment=(
                steps_per_segment
            ),

            condition=jnp.asarray(
                bathymetry,
                dtype=jnp.float32,
            ),

            method="RK4",
        )
    )

    return normalize_batch_prediction(
        prediction,
        n_trajectories=(
            n_trajectories
        ),
        n_times=(
            n_times
        ),
    )


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
        sample_index=(
            sample_index
        ),
    )

    predictions = {}

    for key in MODEL_KEYS:

        print()
        print(
            "Running",
            MODEL_LABELS[key],
            "on",
            case_name,
        )

        (
            hill,
            flat,
        ) = diag.model_pair_rollout(
            methods[key],
            states[key],
            data,

            steps_per_segment=(
                steps_per_segment
            ),
        )

        predictions[key] = {

            "hill":
                hill,

            "flat":
                flat,
        }

    return {

        "data":
            data,

        "models":
            predictions,
    }


# =====================================================================
# DIRECT CONDITION METRICS
# =====================================================================

def calculate_direct_condition(
    result,
    *,
    methods,
    states,
    plot_time,
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

    q_same = data[
        "terrain_q"
    ][
        time_index
    ]

    b = data[
        "b"
    ]

    b_zero = np.zeros_like(
        b
    )

    h = np.maximum(
        q_same[
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

    expected = np.stack(
        [
            expected_hu,
            expected_hv,
        ],
        axis=-1,
    )

    responses = {}

    for key in MODEL_KEYS:

        with_b = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q_same,
                b,
                time_value,
            )
        )

        without_b = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q_same,
                b_zero,
                time_value,
            )
        )

        responses[key] = (
            with_b
            -
            without_b
        )

    metric_rows = [
        {
            "model":
                "PyClaw/SWE reference",

            "direct_rel_l2":
                0.0,

            "direct_cosine":
                1.0,

            "direct_gain":
                1.0,

            "hu_rel_l2":
                0.0,

            "hv_rel_l2":
                0.0,
        }
    ]

    for key in MODEL_KEYS:

        response = responses[
            key
        ]

        response_momentum = np.stack(
            [
                response[
                    ...,
                    1
                ],

                response[
                    ...,
                    2
                ],
            ],
            axis=-1,
        )

        metric_rows.append(
            {
                "model":
                    MODEL_LABELS[key],

                "direct_rel_l2":
                    relative_l2(
                        expected,
                        response_momentum,
                    ),

                "direct_cosine":
                    cosine_similarity(
                        expected,
                        response_momentum,
                    ),

                "direct_gain":
                    gain_ratio(
                        expected,
                        response_momentum,
                    ),

                "hu_rel_l2":
                    relative_l2(
                        expected_hu,
                        response[
                            ...,
                            1
                        ],
                    ),

                "hv_rel_l2":
                    relative_l2(
                        expected_hv,
                        response[
                            ...,
                            2
                        ],
                    ),
            }
        )

    return (
        time_value,
        expected_hu,
        expected_hv,
        responses,
        metric_rows,
    )


# =====================================================================
# PLOT 01
# DIRECT CONDITION
# =====================================================================

def plot_direct_condition(
    result,
    *,
    methods,
    states,
    plot_time,
    output_path,
):

    (
        time_value,
        expected_hu,
        expected_hv,
        responses,
        _,
    ) = calculate_direct_condition(
        result,
        methods=methods,
        states=states,
        plot_time=plot_time,
    )

    b = result[
        "data"
    ][
        "b"
    ]

    column_labels = [
        "PyClaw / SWE reference",
        "Bathy-CFO",
        "Final Geometry-U-FNO",
    ]

    hu_fields = [
        expected_hu,
        responses[
            "cfo"
        ][
            ...,
            1
        ],
        responses[
            "final"
        ][
            ...,
            1
        ],
    ]

    hv_fields = [
        expected_hv,
        responses[
            "cfo"
        ][
            ...,
            2
        ],
        responses[
            "final"
        ][
            ...,
            2
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
            15,
            9,
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
                column_labels[
                    column
                ]
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

            vlim=hv_lim,

            title=(
                column_labels[
                    column
                ]
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
            Line2D(
                [0],
                [0],
                linewidth=2,
                label=(
                    "Bathymetry contours"
                ),
            )
        ],

        loc="upper right",

        frameon=True,
    )

    fig.suptitle(
        (
            "Direct bathymetry response\n"
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
            0.93,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# PLOT 02
# ISOLATED GAUSSIAN HILL EFFECT
# =====================================================================

def plot_isolated_hill_error(
    gaussian_results,
    *,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
    )

    for column, case_name in enumerate(
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

        # PyClaw reference is zero error by definition.
        axes[
            column
        ].plot(
            time,
            np.zeros_like(
                time
            ),
            linestyle="--",
            linewidth=2,
            label="PyClaw reference",
        )

        for key in MODEL_KEYS:

            model = result[
                "models"
            ][key]

            error = (
                diag.relative_effect_error(
                    data[
                        "terrain_q"
                    ],
                    data[
                        "flat_q"
                    ],
                    model[
                        "hill"
                    ],
                    model[
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
                label=MODEL_LABELS[
                    key
                ],
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
                "isolated terrain effect"
            )
        )

        axes[
            column
        ].grid(
            alpha=0.30
        )

        axes[
            column
        ].legend(
            fontsize=8,
            frameon=True,
        )

    fig.suptitle(
        "Error in the isolated Gaussian-hill effect",
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

    plt.close(
        fig
    )


# =====================================================================
# SHARED TERRAIN SPEED GRID
# =====================================================================

def plot_terrain_speed_grid(
    results,
    case_names,
    *,
    plot_time,
    output_path,
    title,
):

    column_labels = [
        "PyClaw",
        "Bathy-CFO",
        "Final Geometry-U-FNO",
    ]

    n_rows = len(
        case_names
    )

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(
            13,
            4.0
            *
            n_rows,
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

        time_index = int(
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
                time_index,
            ),

            diag.terrain_speed_effect(
                result[
                    "models"
                ][
                    "cfo"
                ][
                    "hill"
                ],
                result[
                    "models"
                ][
                    "cfo"
                ][
                    "flat"
                ],
                time_index,
            ),

            diag.terrain_speed_effect(
                result[
                    "models"
                ][
                    "final"
                ][
                    "hill"
                ],
                result[
                    "models"
                ][
                    "final"
                ][
                    "flat"
                ],
                time_index,
            ),
        ]

        limit = max(
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

                vlim=limit,

                title=(
                    case_name
                    +
                    "\n"
                    +
                    column_labels[
                        column
                    ]
                ),

                b=data[
                    "b"
                ],
            )

            colorbar = fig.colorbar(
                image,

                ax=axes[
                    row,
                    column
                ],

                shrink=0.75,
            )

            colorbar.set_label(
                (
                    r"$|\mathbf{u}|_{terrain}"
                    r"-|\mathbf{u}|_{flat}$"
                )
            )

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                linewidth=2,
                label=(
                    "Bathymetry contours"
                ),
            )
        ],

        loc="upper right",

        frameon=True,
    )

    fig.suptitle(
        (
            title
            +
            f" at t={plot_time:.2f}"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# PLOT 05
# CROSS SECTIONS
#
# NO BATHYMETRY DOTTED / GAUSSIAN PROFILE.
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

    b = data[
        "b"
    ]

    (
        ix,
        iy,
    ) = hill_center_indices(
        b
    )

    (
        x,
        y,
    ) = cell_coordinates()

    fields = {

        "PyClaw":
            data[
                "terrain_q"
            ][
                time_index
            ],

        "Bathy-CFO":
            result[
                "models"
            ][
                "cfo"
            ][
                "hill"
            ][
                time_index
            ],

        "Final Geometry-U-FNO":
            result[
                "models"
            ][
                "final"
            ][
                "hill"
            ][
                time_index
            ],
    }

    variables = [
        (
            "h",
            0,
        ),
        (
            "hu",
            1,
        ),
        (
            "hv",
            2,
        ),
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
        variable,
        channel,
    ) in enumerate(
        variables
    ):

        # ---------------------------------------------------------
        # x-section: y = constant
        # ---------------------------------------------------------

        ax = axes[
            0,
            column
        ]

        for label, field in fields.items():

            ax.plot(
                x,

                field[
                    :,
                    iy,
                    channel
                ],

                linewidth=2,

                label=label,
            )

        ax.set_title(
            (
                f"{variable} along "
                f"y={y[iy]:.3f}"
            )
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            variable
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            fontsize=8,
            frameon=True,
        )

        # ---------------------------------------------------------
        # y-section: x = constant
        # ---------------------------------------------------------

        ax = axes[
            1,
            column
        ]

        for label, field in fields.items():

            ax.plot(
                y,

                field[
                    ix,
                    :,
                    channel
                ],

                linewidth=2,

                label=label,
            )

        ax.set_title(
            (
                f"{variable} along "
                f"x={x[ix]:.3f}"
            )
        )

        ax.set_xlabel(
            "y"
        )

        ax.set_ylabel(
            variable
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            fontsize=8,
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

    plt.close(
        fig
    )


# =====================================================================
# PLOT 06
# 3D BATHYMETRY
# =====================================================================

def plot_3d_bathymetry(
    result,
    *,
    output_path,
):

    b = result[
        "data"
    ][
        "b"
    ]

    (
        x,
        y,
    ) = cell_coordinates()

    (
        X,
        Y,
    ) = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    (
        ix,
        iy,
    ) = hill_center_indices(
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

        alpha=0.72,

        linewidth=0,

        antialiased=True,
    )

    y_line, = ax.plot(
        x,

        np.full_like(
            x,
            y[
                iy
            ],
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

    x_line, = ax.plot(
        np.full_like(
            y,
            x[
                ix
            ],
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

    bathy_proxy = Patch(
        facecolor="steelblue",
        alpha=0.72,
        label=(
            "Gaussian bathymetry"
        ),
    )

    ax.legend(
        handles=[
            bathy_proxy,
            y_line,
            x_line,
        ],

        loc="upper right",

        frameon=True,
    )

    ax.set_xlabel(
        "x"
    )

    ax.set_ylabel(
        "y"
    )

    ax.set_zlabel(
        "b"
    )

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

    plt.close(
        fig
    )


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

    (
        x,
        y,
    ) = cell_coordinates()

    (
        X,
        Y,
    ) = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    q_fields = [

        (
            "PyClaw",

            data[
                "terrain_q"
            ][
                time_index
            ],
        ),

        (
            "Bathy-CFO",

            result[
                "models"
            ][
                "cfo"
            ][
                "hill"
            ][
                time_index
            ],
        ),

        (
            "Final Geometry-U-FNO",

            result[
                "models"
            ][
                "final"
            ][
                "hill"
            ][
                time_index
            ],
        ),
    ]

    free_surface_color = (
        "peru"
    )

    bathymetry_color = (
        "lightsteelblue"
    )

    fig = plt.figure(
        figsize=(
            15,
            5,
        )
    )

    for column, (
        label,
        q,
    ) in enumerate(
        q_fields,
        start=1,
    ):

        eta = (
            q[
                ...,
                0
            ]
            +
            b
        )

        ax = fig.add_subplot(
            1,
            3,
            column,
            projection="3d",
        )

        # ---------------------------------------------------------
        # BATHYMETRY — BLUE
        # ---------------------------------------------------------

        ax.plot_surface(
            X,
            Y,
            b,

            color=(
                bathymetry_color
            ),

            alpha=0.50,

            linewidth=0,
        )

        # ---------------------------------------------------------
        # FREE SURFACE — BROWN
        # ---------------------------------------------------------

        ax.plot_surface(
            X,
            Y,
            eta,

            color=(
                free_surface_color
            ),

            alpha=0.82,

            linewidth=0,
        )

        brown_proxy = Patch(
            facecolor=(
                free_surface_color
            ),

            edgecolor="black",

            alpha=0.82,

            label=(
                r"Free surface "
                r"$\eta=h+b$"
            ),
        )

        blue_proxy = Patch(
            facecolor=(
                bathymetry_color
            ),

            edgecolor="black",

            alpha=0.50,

            label=(
                r"Bathymetry $b$"
            ),
        )

        ax.legend(
            handles=[
                brown_proxy,
                blue_proxy,
            ],

            loc="upper right",

            fontsize=8,

            frameon=True,
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

        ax.set_zlabel(
            "height"
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

    plt.close(
        fig
    )


# =====================================================================
# PLOT 08
# 3D h / hu / hv WITH SECTION PATHS
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

    (
        ix,
        iy,
    ) = hill_center_indices(
        b
    )

    (
        x,
        y,
    ) = cell_coordinates()

    (
        X,
        Y,
    ) = np.meshgrid(
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

        "Bathy-CFO":
            result[
                "models"
            ][
                "cfo"
            ][
                "hill"
            ][
                time_index
            ],

        "Final Geometry-U-FNO":
            result[
                "models"
            ][
                "final"
            ][
                "hill"
            ][
                time_index
            ],
    }

    variables = [
        (
            "h",
            0,
        ),
        (
            "hu",
            1,
        ),
        (
            "hv",
            2,
        ),
    ]

    labels = list(
        model_fields.keys()
    )

    # -----------------------------------------------------------------
    # SHARED LIMITS PER VARIABLE
    # -----------------------------------------------------------------

    limits = {}

    for variable, channel in variables:

        all_values = np.concatenate(
            [
                model_fields[
                    label
                ][
                    ...,
                    channel
                ].ravel()

                for label
                in labels
            ]
        )

        low = float(
            np.min(
                all_values
            )
        )

        high = float(
            np.max(
                all_values
            )
        )

        span = max(
            high
            -
            low,
            1.0e-6,
        )

        margin = (
            0.08
            *
            span
        )

        limits[
            variable
        ] = (
            low
            -
            margin,
            high
            +
            margin,
        )

    fig = plt.figure(
        figsize=(
            15,
            13,
        )
    )

    for row, (
        variable,
        channel,
    ) in enumerate(
        variables
    ):

        (
            z_low,
            z_high,
        ) = limits[
            variable
        ]

        z_span = max(
            z_high
            -
            z_low,
            1.0e-6,
        )

        floor = (
            z_low
            -
            0.06
            *
            z_span
        )

        lift = (
            0.015
            *
            z_span
        )

        for column, label in enumerate(
            labels
        ):

            field = model_fields[
                label
            ][
                ...,
                channel
            ]

            subplot_number = (
                row
                *
                3
                +
                column
                +
                1
            )

            ax = fig.add_subplot(
                3,
                3,
                subplot_number,

                projection="3d",
            )

            # -----------------------------------------------------
            # STATE SURFACE
            # -----------------------------------------------------

            ax.plot_surface(
                X,
                Y,
                field,

                color="steelblue",

                alpha=0.62,

                linewidth=0,

                antialiased=True,
            )

            # -----------------------------------------------------
            # y = constant SECTION
            # -----------------------------------------------------

            y_section, = ax.plot(
                x,

                np.full_like(
                    x,
                    y[
                        iy
                    ],
                ),

                field[
                    :,
                    iy
                ]
                +
                lift,

                linewidth=3,

                color="darkorange",

                label=(
                    f"y={y[iy]:.3f} section"
                ),
            )

            # -----------------------------------------------------
            # x = constant SECTION
            # -----------------------------------------------------

            x_section, = ax.plot(
                np.full_like(
                    y,
                    x[
                        ix
                    ],
                ),

                y,

                field[
                    ix,
                    :
                ]
                +
                lift,

                linewidth=3,

                linestyle="--",

                color="green",

                label=(
                    f"x={x[ix]:.3f} section"
                ),
            )

            # -----------------------------------------------------
            # BATHYMETRY CONTOURS
            # -----------------------------------------------------

            bmax = float(
                np.max(
                    b
                )
            )

            if bmax > 1.0e-12:

                positive_b = b[
                    b
                    >
                    0.0
                ]

                if (
                    positive_b.size
                    >
                    0
                ):

                    levels = np.linspace(
                        max(
                            float(
                                np.min(
                                    positive_b
                                )
                            ),
                            0.10
                            *
                            bmax,
                        ),

                        bmax,

                        6,
                    )

                    ax.contour(
                        X,
                        Y,
                        b,

                        levels=levels,

                        zdir="z",

                        offset=floor,

                        colors="purple",

                        linewidths=1.0,
                    )

            surface_proxy = Patch(
                facecolor="steelblue",

                alpha=0.62,

                label=(
                    f"{variable}(x,y)"
                ),
            )

            bathy_proxy = Line2D(
                [0],
                [0],

                color="purple",

                linewidth=1.5,

                label=(
                    "Bathymetry contours"
                ),
            )

            ax.legend(
                handles=[
                    surface_proxy,
                    y_section,
                    x_section,
                    bathy_proxy,
                ],

                loc="upper right",

                fontsize=7,

                frameon=True,
            )

            if row == 0:

                ax.set_title(
                    label,
                    fontsize=10,
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
                variable,
                fontsize=8,
            )

            ax.set_zlim(
                floor,
                z_high,
            )

            ax.view_init(
                elev=28,
                azim=-58,
            )

            ax.tick_params(
                labelsize=7
            )

    fig.suptitle(
        (
            "3D h, hu, hv fields through Gaussian hill\n"
            f"t={time_value:.2f}, "
            f"y-section={y[iy]:.3f}, "
            f"x-section={x[ix]:.3f}"
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
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# AGGREGATE ROWS
# =====================================================================

def average_rows(
    rows,
    *,
    model_name,
    numeric_columns,
):

    selected = [
        row
        for row
        in rows
        if (
            row[
                "model"
            ]
            ==
            model_name
        )
    ]

    output = {
        "model":
            model_name,
    }

    for column in numeric_columns:

        values = [
            float(
                row[
                    column
                ]
            )

            for row
            in selected
        ]

        output[
            column
        ] = float(
            np.mean(
                values
            )
        )

    return output


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    configure_diag()

    output_dir = resolve_path(
        args.output_dir
    )

    figure_dir = (
        output_dir
        /
        "figures"
    )

    table_dir = (
        output_dir
        /
        "tables"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # BUILD MODELS
    # =================================================================

    (
        methods,
        states,
    ) = build_models(
        args
    )

    # =================================================================
    # TABLE 00 — MODEL CONFIGURATION
    # =================================================================

    configuration_rows = [

        {
            "model":
                "PyClaw",

            "backbone":
                "Finite-volume SWE solver",

            "geometry_input":
                "Bathymetry b",

            "lambda_CFO":
                0.0,

            "lambda_PDE":
                0.0,

            "lambda_bed":
                0.0,

            "lambda_WB":
                0.0,

            "lambda_tangent":
                0.0,
        },

        {
            "model":
                "Bathy-CFO",

            "backbone":
                "FNO2d",

            "geometry_input":
                "Raw b as condition",

            "lambda_CFO":
                1.0,

            "lambda_PDE":
                0.0,

            "lambda_bed":
                0.0,

            "lambda_WB":
                0.0,

            "lambda_tangent":
                0.0,
        },

        {
            "model":
                "Final Geometry-U-FNO",

            "backbone":
                "Geometry-U-FNO",

            "geometry_input":
                "[b, b_x, b_y] -> GeometryEncoder2d",

            "lambda_CFO":
                1.0,

            "lambda_PDE":
                args.lambda_pde,

            "lambda_bed":
                args.lambda_bed,

            "lambda_WB":
                args.lambda_wb,

            "lambda_tangent":
                0.0,
        },
    ]

    write_table(
        configuration_rows,

        columns=[
            "model",
            "backbone",
            "geometry_input",
            "lambda_CFO",
            "lambda_PDE",
            "lambda_bed",
            "lambda_WB",
            "lambda_tangent",
        ],

        csv_path=(
            table_dir
            /
            "table_00_model_configuration.csv"
        ),

        markdown_path=(
            table_dir
            /
            "table_00_model_configuration.md"
        ),

        title=(
            "Model Configuration"
        ),
    )

    # =================================================================
    # STANDARD ID DATASET
    # =================================================================

    id_dataset_path = resolve_path(
        args.id_data
    )

    id_dataset = (
        load_bathymetry_dataset(
            id_dataset_path
        )
    )

    id_split = require_split(
        id_dataset,
        "test_id",
    )

    id_truth = np.asarray(
        id_split[
            "q"
        ],
        dtype=np.float32,
    )

    id_bathymetry = np.asarray(
        id_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    id_predictions = {}

    for key in MODEL_KEYS:

        print()
        print(
            "Running standard ID rollout:",
            MODEL_LABELS[key],
        )

        id_predictions[
            key
        ] = run_id_rollout(
            methods[key],
            states[key],

            id_truth,
            id_bathymetry,

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

    # -----------------------------------------------------------------
    # ID TABLE
    # -----------------------------------------------------------------

    id_rows = []

    reference_metrics = rollout_metrics(
        id_truth,
        id_truth,
        id_bathymetry,
    )

    id_rows.append(
        {
            "model":
                "PyClaw",

            **reference_metrics,
        }
    )

    for key in MODEL_KEYS:

        metrics = rollout_metrics(
            id_truth,
            id_predictions[key],
            id_bathymetry,
        )

        id_rows.append(
            {
                "model":
                    MODEL_LABELS[key],

                **metrics,
            }
        )

    id_columns = [
        "model",
        "rel_l2",
        "rmse",
        "rel_fro",
        "mean_time_rel_l2",
        "final_time_rel_l2",
        "h_rel_l2",
        "hu_rel_l2",
        "hv_rel_l2",
        "momentum_rel_l2",
        "eta_rel_l2",
        "mean_mass_drift",
        "final_mass_drift",
        "max_mass_drift",
        "mean_mass_error_vs_pyclaw",
        "final_mass_error_vs_pyclaw",
        "max_mass_error_vs_pyclaw",
        "minimum_depth",
        "negative_depth_fraction",
    ]

    write_table(
        id_rows,

        columns=id_columns,

        csv_path=(
            table_dir
            /
            "table_01_id_metrics.csv"
        ),

        markdown_path=(
            table_dir
            /
            "table_01_id_metrics.md"
        ),

        title=(
            "Standard 32x32 ID Rollout Metrics"
        ),
    )

    # =================================================================
    # LOAD COUNTERFACTUAL / OOD DATA
    # =================================================================

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
        # GAUSSIAN RESULTS
        # =============================================================

        gaussian_results = {}

        for case_name in GAUSSIAN_CASES:

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
        # OOD RESULTS
        # =============================================================

        ood_results = {}

        for case_name in OOD_CASES:

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
        # SELECT PLOT TIME
        # =============================================================

        reference_time = (
            gaussian_results[
                GAUSSIAN_CASES[0]
            ][
                "data"
            ][
                "time"
            ]
        )

        terrain_time_index = int(
            np.argmin(
                np.abs(
                    reference_time
                    -
                    args.plot_time
                )
            )
        )

        # =============================================================
        # TABLE 02
        # GAUSSIAN ROLLOUT METRICS
        # =============================================================

        gaussian_rollout_rows = []

        # -------------------------------------------------------------
        # FLAT CASE ONCE
        # -------------------------------------------------------------

        flat_reference = (
            gaussian_results[
                GAUSSIAN_CASES[0]
            ]
        )

        flat_truth = (
            flat_reference[
                "data"
            ][
                "flat_q"
            ]
        )

        flat_b = np.zeros(
            (
                RESOLUTION,
                RESOLUTION,
            ),
            dtype=np.float32,
        )

        gaussian_rollout_rows.append(
            {
                "case":
                    "flat",

                "model":
                    "PyClaw",

                **rollout_metrics(
                    flat_truth,
                    flat_truth,
                    flat_b,
                ),
            }
        )

        for key in MODEL_KEYS:

            gaussian_rollout_rows.append(
                {
                    "case":
                        "flat",

                    "model":
                        MODEL_LABELS[key],

                    **rollout_metrics(
                        flat_truth,

                        flat_reference[
                            "models"
                        ][key][
                            "flat"
                        ],

                        flat_b,
                    ),
                }
            )

        # -------------------------------------------------------------
        # THREE HILLS
        # -------------------------------------------------------------

        for case_name in GAUSSIAN_CASES:

            result = gaussian_results[
                case_name
            ]

            true_hill = result[
                "data"
            ][
                "terrain_q"
            ]

            b = result[
                "data"
            ][
                "b"
            ]

            gaussian_rollout_rows.append(
                {
                    "case":
                        case_name,

                    "model":
                        "PyClaw",

                    **rollout_metrics(
                        true_hill,
                        true_hill,
                        b,
                    ),
                }
            )

            for key in MODEL_KEYS:

                gaussian_rollout_rows.append(
                    {
                        "case":
                            case_name,

                        "model":
                            MODEL_LABELS[key],

                        **rollout_metrics(
                            true_hill,

                            result[
                                "models"
                            ][key][
                                "hill"
                            ],

                            b,
                        ),
                    }
                )

        gaussian_columns = [
            "case",
            "model",
            "rel_l2",
            "rmse",
            "rel_fro",
            "mean_time_rel_l2",
            "final_time_rel_l2",
            "h_rel_l2",
            "hu_rel_l2",
            "hv_rel_l2",
            "momentum_rel_l2",
            "eta_rel_l2",
            "mean_mass_drift",
            "final_mass_drift",
            "max_mass_drift",
            "mean_mass_error_vs_pyclaw",
            "final_mass_error_vs_pyclaw",
            "minimum_depth",
            "negative_depth_fraction",
        ]

        write_table(
            gaussian_rollout_rows,

            columns=gaussian_columns,

            csv_path=(
                table_dir
                /
                "table_02_gaussian_rollout_metrics.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_02_gaussian_rollout_metrics.md"
            ),

            title=(
                "Gaussian Counterfactual Rollout Metrics"
            ),
        )

        # =============================================================
        # TABLE 03
        # MASS DRIFT
        # =============================================================

        mass_rows = []

        for row in gaussian_rollout_rows:

            mass_rows.append(
                {
                    "case":
                        row[
                            "case"
                        ],

                    "model":
                        row[
                            "model"
                        ],

                    "mean_mass_drift":
                        row[
                            "mean_mass_drift"
                        ],

                    "final_mass_drift":
                        row[
                            "final_mass_drift"
                        ],

                    "max_mass_drift":
                        row[
                            "max_mass_drift"
                        ],

                    "mean_mass_error_vs_pyclaw":
                        row[
                            "mean_mass_error_vs_pyclaw"
                        ],

                    "final_mass_error_vs_pyclaw":
                        row[
                            "final_mass_error_vs_pyclaw"
                        ],

                    "max_mass_error_vs_pyclaw":
                        row[
                            "max_mass_error_vs_pyclaw"
                        ],
                }
            )

        write_table(
            mass_rows,

            columns=[
                "case",
                "model",
                "mean_mass_drift",
                "final_mass_drift",
                "max_mass_drift",
                "mean_mass_error_vs_pyclaw",
                "final_mass_error_vs_pyclaw",
                "max_mass_error_vs_pyclaw",
            ],

            csv_path=(
                table_dir
                /
                "table_03_gaussian_mass_drift.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_03_gaussian_mass_drift.md"
            ),

            title=(
                "Gaussian Counterfactual Mass Conservation"
            ),
        )

        # =============================================================
        # TABLE 04
        # GAUSSIAN TERRAIN EFFECT
        # =============================================================

        gaussian_effect_rows = []

        for case_name in GAUSSIAN_CASES:

            result = gaussian_results[
                case_name
            ]

            data = result[
                "data"
            ]

            gaussian_effect_rows.append(
                {
                    "case":
                        case_name,

                    "model":
                        "PyClaw",

                    "effect_rel_l2":
                        0.0,

                    "mean_effect_error":
                        0.0,

                    "final_effect_error":
                        0.0,

                    "speed_effect_rel_l2":
                        0.0,

                    "speed_effect_cosine":
                        1.0,

                    "speed_effect_gain":
                        1.0,
                }
            )

            for key in MODEL_KEYS:

                metrics = (
                    terrain_effect_metrics(
                        data[
                            "terrain_q"
                        ],

                        data[
                            "flat_q"
                        ],

                        result[
                            "models"
                        ][key][
                            "hill"
                        ],

                        result[
                            "models"
                        ][key][
                            "flat"
                        ],

                        time_index=(
                            terrain_time_index
                        ),
                    )
                )

                gaussian_effect_rows.append(
                    {
                        "case":
                            case_name,

                        "model":
                            MODEL_LABELS[key],

                        **metrics,
                    }
                )

        write_table(
            gaussian_effect_rows,

            columns=[
                "case",
                "model",
                "effect_rel_l2",
                "mean_effect_error",
                "final_effect_error",
                "speed_effect_rel_l2",
                "speed_effect_cosine",
                "speed_effect_gain",
            ],

            csv_path=(
                table_dir
                /
                "table_04_gaussian_terrain_effect.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_04_gaussian_terrain_effect.md"
            ),

            title=(
                "Gaussian Isolated Terrain-Effect Metrics"
            ),
        )

        # =============================================================
        # TABLE 05
        # DIRECT CONDITION
        # =============================================================

        (
            _,
            _,
            _,
            _,
            direct_rows,
        ) = calculate_direct_condition(
            gaussian_results[
                args.direct_case
            ],

            methods=methods,
            states=states,

            plot_time=(
                args.plot_time
            ),
        )

        write_table(
            direct_rows,

            columns=[
                "model",
                "direct_rel_l2",
                "direct_cosine",
                "direct_gain",
                "hu_rel_l2",
                "hv_rel_l2",
            ],

            csv_path=(
                table_dir
                /
                "table_05_direct_condition_metrics.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_05_direct_condition_metrics.md"
            ),

            title=(
                "Direct Bathymetry-Response Metrics"
            ),
        )

        # =============================================================
        # TABLE 06
        # OOD ROLLOUT
        # =============================================================

        ood_rollout_rows = []

        for case_name in OOD_CASES:

            result = ood_results[
                case_name
            ]

            true_hill = result[
                "data"
            ][
                "terrain_q"
            ]

            b = result[
                "data"
            ][
                "b"
            ]

            ood_rollout_rows.append(
                {
                    "case":
                        case_name,

                    "model":
                        "PyClaw",

                    **rollout_metrics(
                        true_hill,
                        true_hill,
                        b,
                    ),
                }
            )

            for key in MODEL_KEYS:

                ood_rollout_rows.append(
                    {
                        "case":
                            case_name,

                        "model":
                            MODEL_LABELS[key],

                        **rollout_metrics(
                            true_hill,

                            result[
                                "models"
                            ][key][
                                "hill"
                            ],

                            b,
                        ),
                    }
                )

        write_table(
            ood_rollout_rows,

            columns=gaussian_columns,

            csv_path=(
                table_dir
                /
                "table_06_ood_rollout_metrics.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_06_ood_rollout_metrics.md"
            ),

            title=(
                "Unseen-Bathymetry Rollout Metrics"
            ),
        )

        # =============================================================
        # TABLE 07
        # OOD TERRAIN EFFECT
        # =============================================================

        ood_effect_rows = []

        for case_name in OOD_CASES:

            result = ood_results[
                case_name
            ]

            data = result[
                "data"
            ]

            ood_time_index = int(
                np.argmin(
                    np.abs(
                        data[
                            "time"
                        ]
                        -
                        args.plot_time
                    )
                )
            )

            ood_effect_rows.append(
                {
                    "case":
                        case_name,

                    "model":
                        "PyClaw",

                    "effect_rel_l2":
                        0.0,

                    "mean_effect_error":
                        0.0,

                    "final_effect_error":
                        0.0,

                    "speed_effect_rel_l2":
                        0.0,

                    "speed_effect_cosine":
                        1.0,

                    "speed_effect_gain":
                        1.0,
                }
            )

            for key in MODEL_KEYS:

                metrics = (
                    terrain_effect_metrics(
                        data[
                            "terrain_q"
                        ],

                        data[
                            "flat_q"
                        ],

                        result[
                            "models"
                        ][key][
                            "hill"
                        ],

                        result[
                            "models"
                        ][key][
                            "flat"
                        ],

                        time_index=(
                            ood_time_index
                        ),
                    )
                )

                ood_effect_rows.append(
                    {
                        "case":
                            case_name,

                        "model":
                            MODEL_LABELS[key],

                        **metrics,
                    }
                )

        write_table(
            ood_effect_rows,

            columns=[
                "case",
                "model",
                "effect_rel_l2",
                "mean_effect_error",
                "final_effect_error",
                "speed_effect_rel_l2",
                "speed_effect_cosine",
                "speed_effect_gain",
            ],

            csv_path=(
                table_dir
                /
                "table_07_ood_terrain_effect.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_07_ood_terrain_effect.md"
            ),

            title=(
                "Unseen-Bathymetry Terrain-Effect Metrics"
            ),
        )

        # =============================================================
        # TABLE 08
        # FINAL COMPACT SUMMARY
        # =============================================================

        summary_rows = []

        direct_lookup = {
            row[
                "model"
            ]:
                row

            for row
            in direct_rows
        }

        for model_name in [
            "PyClaw",
            "Bathy-CFO",
            "Final Geometry-U-FNO",
        ]:

            # ---------------------------------------------------------
            # ID
            # ---------------------------------------------------------

            id_row = next(
                row
                for row
                in id_rows
                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name
                )
            )

            # ---------------------------------------------------------
            # Gaussian — exclude flat
            # ---------------------------------------------------------

            gaussian_selected = [
                row

                for row
                in gaussian_rollout_rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name

                    and

                    row[
                        "case"
                    ]
                    !=
                    "flat"
                )
            ]

            gaussian_effect_selected = [
                row

                for row
                in gaussian_effect_rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name
                )
            ]

            # ---------------------------------------------------------
            # OOD
            # ---------------------------------------------------------

            ood_selected = [
                row

                for row
                in ood_rollout_rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name
                )
            ]

            ood_effect_selected = [
                row

                for row
                in ood_effect_rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    model_name
                )
            ]

            direct_name = (
                "PyClaw/SWE reference"
                if (
                    model_name
                    ==
                    "PyClaw"
                )
                else
                model_name
            )

            direct_row = (
                direct_lookup[
                    direct_name
                ]
            )

            summary_rows.append(
                {
                    "model":
                        model_name,

                    "id_rel_l2":
                        id_row[
                            "rel_l2"
                        ],

                    "id_rmse":
                        id_row[
                            "rmse"
                        ],

                    "id_eta_rel_l2":
                        id_row[
                            "eta_rel_l2"
                        ],

                    "id_final_mass_drift":
                        id_row[
                            "final_mass_drift"
                        ],

                    "gaussian_mean_rel_l2":
                        float(
                            np.mean(
                                [
                                    row[
                                        "rel_l2"
                                    ]

                                    for row
                                    in gaussian_selected
                                ]
                            )
                        ),

                    "gaussian_mean_final_mass_drift":
                        float(
                            np.mean(
                                [
                                    row[
                                        "final_mass_drift"
                                    ]

                                    for row
                                    in gaussian_selected
                                ]
                            )
                        ),

                    "gaussian_mean_effect_rel_l2":
                        float(
                            np.mean(
                                [
                                    row[
                                        "effect_rel_l2"
                                    ]

                                    for row
                                    in gaussian_effect_selected
                                ]
                            )
                        ),

                    "ood_mean_rel_l2":
                        float(
                            np.mean(
                                [
                                    row[
                                        "rel_l2"
                                    ]

                                    for row
                                    in ood_selected
                                ]
                            )
                        ),

                    "ood_mean_effect_rel_l2":
                        float(
                            np.mean(
                                [
                                    row[
                                        "effect_rel_l2"
                                    ]

                                    for row
                                    in ood_effect_selected
                                ]
                            )
                        ),

                    "direct_response_rel_l2":
                        direct_row[
                            "direct_rel_l2"
                        ],

                    "direct_response_cosine":
                        direct_row[
                            "direct_cosine"
                        ],

                    "direct_response_gain":
                        direct_row[
                            "direct_gain"
                        ],
                }
            )

        write_table(
            summary_rows,

            columns=[
                "model",
                "id_rel_l2",
                "id_rmse",
                "id_eta_rel_l2",
                "id_final_mass_drift",
                "gaussian_mean_rel_l2",
                "gaussian_mean_final_mass_drift",
                "gaussian_mean_effect_rel_l2",
                "ood_mean_rel_l2",
                "ood_mean_effect_rel_l2",
                "direct_response_rel_l2",
                "direct_response_cosine",
                "direct_response_gain",
            ],

            csv_path=(
                table_dir
                /
                "table_08_final_summary.csv"
            ),

            markdown_path=(
                table_dir
                /
                "table_08_final_summary.md"
            ),

            title=(
                "Final PyClaw vs CFO vs Geometry-U-FNO Summary"
            ),
        )

        # =============================================================
        # FIGURE 01
        # =============================================================

        print()
        print(
            "Creating Figure 01..."
        )

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
                figure_dir
                /
                "01_direct_condition_test.png"
            ),
        )

        # =============================================================
        # FIGURE 02
        # =============================================================

        print()
        print(
            "Creating Figure 02..."
        )

        plot_isolated_hill_error(
            gaussian_results,

            output_path=(
                figure_dir
                /
                "02_isolated_hill_effect_error.png"
            ),
        )

        # =============================================================
        # FIGURE 03
        # =============================================================

        print()
        print(
            "Creating Figure 03..."
        )

        plot_terrain_speed_grid(
            gaussian_results,
            GAUSSIAN_CASES,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                figure_dir
                /
                "03_terrain_induced_speed_gaussian.png"
            ),

            title=(
                "Terrain-induced speed "
                "for Gaussian hills"
            ),
        )

        # =============================================================
        # FIGURE 04
        # =============================================================

        print()
        print(
            "Creating Figure 04..."
        )

        plot_terrain_speed_grid(
            ood_results,
            OOD_CASES,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                figure_dir
                /
                "04_terrain_induced_speed_unseen.png"
            ),

            title=(
                "Terrain-induced speed "
                "on unseen bathymetry"
            ),
        )

        # =============================================================
        # COMMON CROSS SECTION RESULT
        # =============================================================

        cross_result = (
            gaussian_results[
                args.cross_section_case
            ]
        )

        # =============================================================
        # FIGURE 05
        # =============================================================

        print()
        print(
            "Creating Figure 05..."
        )

        plot_cross_sections(
            cross_result,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                figure_dir
                /
                "05_gaussian_cross_sections_h_hu_hv.png"
            ),
        )

        # =============================================================
        # FIGURE 06
        # =============================================================

        print()
        print(
            "Creating Figure 06..."
        )

        plot_3d_bathymetry(
            cross_result,

            output_path=(
                figure_dir
                /
                "06_gaussian_3d_bathymetry_cross_sections.png"
            ),
        )

        # =============================================================
        # FIGURE 07
        # =============================================================

        print()
        print(
            "Creating Figure 07..."
        )

        plot_3d_free_surface(
            cross_result,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                figure_dir
                /
                "07_gaussian_3d_free_surface.png"
            ),
        )

        # =============================================================
        # FIGURE 08
        # =============================================================

        print()
        print(
            "Creating Figure 08..."
        )

        plot_3d_state_cross_sections(
            cross_result,

            plot_time=(
                args.plot_time
            ),

            output_path=(
                figure_dir
                /
                "08_gaussian_3d_h_hu_hv_cross_sections.png"
            ),
        )

    finally:

        gaussian_archive.close()

        ood_archive.close()

    # =================================================================
    # COMPLETE
    # =================================================================

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL CFO VS GEOMETRY-U-FNO ANALYSIS COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        "Figures:"
    )

    print(
        figure_dir
    )

    print()

    print(
        "Tables:"
    )

    print(
        table_dir
    )


if __name__ == "__main__":

    main()