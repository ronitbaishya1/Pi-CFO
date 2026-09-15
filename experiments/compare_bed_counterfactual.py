"""
Compare original PI-CFO against bed-regularized PI-CFO
using the controlled Gaussian counterfactual experiment.

We compare:

    PyClaw reference
    Original PI-CFO
    Bed-regularized PI-CFO

for:

    flat
    hill_left
    hill_center
    hill_right

The goal is to determine whether adding L_bed improves not only
the instantaneous terrain response, but also the complete rollout.

Outputs
-------
1. Terrain-induced speed comparison
2. Terrain-effect error versus time
3. Direct bed-response comparison
4. counterfactual_metrics.csv

Important metrics
-----------------
delta_q_error:
    Error in the isolated terrain effect

        q_hill - q_flat

velocity_cosine:
    Direction/spatial-pattern agreement of terrain-induced velocity

velocity_magnitude_ratio:
    1.0 = ideal
    >1  = terrain effect too strong
    <1  = terrain effect too weak
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# PROJECT ROOT
# ================================================================

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


# ================================================================
# PROJECT IMPORTS
# ================================================================

from bathy_pi_cfo import (
    BathymetryPhysicsInformedCFO,
)

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from models.factory import (
    build_model,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)

from utils.physics_swe_bathy import (
    well_balanced_bed_source,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_32.h5"
        ),
    )

    parser.add_argument(
        "--bed-lambda",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--bed-ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pi-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_pi_lam003/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--time",
        type=float,
        default=0.50,
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

    parser.add_argument(
        "--dx",
        type=float,
        default=0.15625,
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=0.15625,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "bed_counterfactual_comparison"
        ),
    )

    return parser.parse_args()


# ================================================================
# PATH
# ================================================================

def resolve_path(
    value,
):

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# ================================================================
# LOAD COUNTERFACTUAL DATASET
# ================================================================

def load_dataset(
    path,
):

    output = {}

    with h5py.File(
        path,
        "r",
    ) as h5:

        output[
            "x"
        ] = np.asarray(
            h5[
                "x"
            ],
            dtype=np.float32,
        )

        output[
            "y"
        ] = np.asarray(
            h5[
                "y"
            ],
            dtype=np.float32,
        )

        output[
            "time"
        ] = np.asarray(
            h5[
                "time"
            ],
            dtype=np.float32,
        )

        output[
            "cases"
        ] = {}

        for case_name in [
            "flat",
            "hill_left",
            "hill_center",
            "hill_right",
        ]:

            group = (
                h5[
                    "cases"
                ][
                    case_name
                ]
            )

            output[
                "cases"
            ][
                case_name
            ] = {
                "q":
                    np.asarray(
                        group[
                            "q"
                        ],
                        dtype=np.float32,
                    ),

                "bathymetry":
                    np.asarray(
                        group[
                            "bathymetry"
                        ],
                        dtype=np.float32,
                    ),
            }

    return output


# ================================================================
# VELOCITY
# ================================================================

def velocity(
    q,
):

    h = np.maximum(
        q[
            ...,
            0
        ],
        1e-6,
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

    speed = np.sqrt(
        u**2
        +
        v**2
    )

    return (
        u,
        v,
        speed,
    )


# ================================================================
# COSINE SIMILARITY
# ================================================================

def vector_cosine(
    u_true,
    v_true,
    u_pred,
    v_pred,
):

    true_vector = np.concatenate(
        [
            u_true.ravel(),
            v_true.ravel(),
        ]
    )

    pred_vector = np.concatenate(
        [
            u_pred.ravel(),
            v_pred.ravel(),
        ]
    )

    denominator = (
        np.linalg.norm(
            true_vector
        )
        *
        np.linalg.norm(
            pred_vector
        )
    )

    return float(
        np.dot(
            true_vector,
            pred_vector,
        )
        /
        max(
            denominator,
            1e-12,
        )
    )


# ================================================================
# RELATIVE ERROR
# ================================================================

def relative_error(
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
            1e-12,
        )
    )


# ================================================================
# MAGNITUDE RATIO
# ================================================================

def norm_ratio(
    truth,
    prediction,
):

    return float(
        np.linalg.norm(
            prediction.ravel()
        )
        /
        max(
            np.linalg.norm(
                truth.ravel()
            ),
            1e-12,
        )
    )


# ================================================================
# ORIGINAL PI-CFO
# ================================================================

def build_original_pi(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return BathymetryPhysicsInformedCFO(
        model=model,
        input_shape=input_shape,
        condition_shape=(
            condition_shape
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=0.03,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )


# ================================================================
# BED-REGULARIZED PI-CFO
# ================================================================

def build_bed_pi(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return BathymetryBedRegularizedPICFO(
        model=model,
        input_shape=input_shape,
        condition_shape=(
            condition_shape
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=0.03,
        lambda_bed=(
            args.bed_lambda
        ),
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )


# ================================================================
# RESTORE CHECKPOINT
# ================================================================

def restore_state(
    method,
    directory,
    prefix,
):

    target = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
    )

    state = load_train_state(
        target,
        str(
            directory
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )

    return state


# ================================================================
# RUN MODEL ON ALL TERRAIN CASES
# ================================================================

def run_cases(
    method,
    state,
    cases,
):

    output = {}

    for (
        case_name,
        case,
    ) in cases.items():

        print(
            f"  inference: {case_name}"
        )

        true_q = case[
            "q"
        ]

        bathymetry = case[
            "bathymetry"
        ]

        prediction = (
            method.uniform_inference(
                state,
                true_q[
                    0
                ][
                    None,
                    ...
                ],
                trajectory_points_num=(
                    true_q.shape[
                        0
                    ]
                ),
                steps_per_segment=2,
                condition=(
                    bathymetry[
                        None,
                        ...
                    ]
                ),
                method="RK4",
            )
        )

        output[
            case_name
        ] = np.asarray(
            prediction[
                0
            ],
            dtype=np.float32,
        )

    return output


# ================================================================
# TIME INDEX
# ================================================================

def nearest_time(
    time,
    target,
):

    return int(
        np.argmin(
            np.abs(
                time
                -
                target
            )
        )
    )


# ================================================================
# GRID
# ================================================================

def make_grid(
    x,
    y,
):

    return np.meshgrid(
        x,
        y,
        indexing="ij",
    )


# ================================================================
# BATHYMETRY CONTOURS
# ================================================================

def add_contours(
    ax,
    X,
    Y,
    bathymetry,
):

    maximum = float(
        np.max(
            bathymetry
        )
    )

    if maximum <= 0:

        return

    levels = np.linspace(
        0.2
        *
        maximum,
        0.9
        *
        maximum,
        5,
    )

    ax.contour(
        X,
        Y,
        bathymetry,
        levels=levels,
        linewidths=0.8,
    )


# ================================================================
# PLOT 1
# TERRAIN-INDUCED SPEED
# ================================================================

def plot_speed_effect(
    data,
    pi_prediction,
    bed_prediction,
    time_index,
    output_dir,
):

    cases = data[
        "cases"
    ]

    X, Y = make_grid(
        data[
            "x"
        ],
        data[
            "y"
        ],
    )

    true_flat_speed = velocity(
        cases[
            "flat"
        ][
            "q"
        ][
            time_index
        ]
    )[
        2
    ]

    pi_flat_speed = velocity(
        pi_prediction[
            "flat"
        ][
            time_index
        ]
    )[
        2
    ]

    bed_flat_speed = velocity(
        bed_prediction[
            "flat"
        ][
            time_index
        ]
    )[
        2
    ]

    hill_cases = [
        "hill_left",
        "hill_center",
        "hill_right",
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            14,
            13,
        ),
        constrained_layout=True,
    )

    column_titles = [
        "PyClaw",
        "Original PI-CFO",
        "Bed-PI-CFO",
    ]

    for (
        row,
        case_name,
    ) in enumerate(
        hill_cases
    ):

        true_speed = velocity(
            cases[
                case_name
            ][
                "q"
            ][
                time_index
            ]
        )[
            2
        ]

        pi_speed = velocity(
            pi_prediction[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        bed_speed = velocity(
            bed_prediction[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        effects = [
            true_speed
            -
            true_flat_speed,

            pi_speed
            -
            pi_flat_speed,

            bed_speed
            -
            bed_flat_speed,
        ]

        maximum = max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            )
            for field
            in effects
        )

        bathymetry = cases[
            case_name
        ][
            "bathymetry"
        ][
            ...,
            0
        ]

        for column in range(
            3
        ):

            ax = axes[
                row,
                column,
            ]

            mesh = ax.pcolormesh(
                X,
                Y,
                effects[
                    column
                ],
                shading="auto",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )

            add_contours(
                ax,
                X,
                Y,
                bathymetry,
            )

            fig.colorbar(
                mesh,
                ax=ax,
                label=(
                    "|u| hill - |u| flat"
                ),
            )

            ax.set_title(
                f"{case_name}\n"
                f"{column_titles[column]}"
            )

            ax.set_xlabel(
                "x"
            )

            ax.set_ylabel(
                "y"
            )

            ax.set_aspect(
                "equal"
            )

    fig.suptitle(
        "Terrain-induced speed: "
        "does L_bed improve the full rollout?"
    )

    fig.savefig(
        output_dir
        /
        "01_terrain_induced_speed_comparison.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# TERRAIN EFFECT ERROR THROUGH TIME
# ================================================================

def error_series(
    truth_flat,
    truth_hill,
    pred_flat,
    pred_hill,
):

    truth_delta = (
        truth_hill
        -
        truth_flat
    )

    pred_delta = (
        pred_hill
        -
        pred_flat
    )

    values = []

    for t_index in range(
        truth_delta.shape[
            0
        ]
    ):

        denominator = np.linalg.norm(
            truth_delta[
                t_index
            ].ravel()
        )

        if denominator < 1e-10:

            values.append(
                0.0
            )

        else:

            values.append(
                relative_error(
                    truth_delta[
                        t_index
                    ],
                    pred_delta[
                        t_index
                    ],
                )
            )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# ================================================================
# PLOT 2
# TERRAIN EFFECT ERROR VS TIME
# ================================================================

def plot_error_vs_time(
    data,
    pi_prediction,
    bed_prediction,
    output_dir,
):

    cases = data[
        "cases"
    ]

    time = data[
        "time"
    ]

    hill_cases = [
        "hill_left",
        "hill_center",
        "hill_right",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            4.5,
        ),
        constrained_layout=True,
    )

    for (
        ax,
        case_name,
    ) in zip(
        axes,
        hill_cases,
    ):

        pi_error = error_series(
            cases[
                "flat"
            ][
                "q"
            ],
            cases[
                case_name
            ][
                "q"
            ],
            pi_prediction[
                "flat"
            ],
            pi_prediction[
                case_name
            ],
        )

        bed_error = error_series(
            cases[
                "flat"
            ][
                "q"
            ],
            cases[
                case_name
            ][
                "q"
            ],
            bed_prediction[
                "flat"
            ],
            bed_prediction[
                case_name
            ],
        )

        ax.plot(
            time,
            pi_error,
            label="Original PI-CFO",
        )

        ax.plot(
            time,
            bed_error,
            label="Bed-PI-CFO",
        )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Relative terrain-effect error"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

    fig.suptitle(
        "Error in the isolated hill effect"
    )

    fig.savefig(
        output_dir
        /
        "02_terrain_effect_error_vs_time.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# DIRECT NETWORK RESPONSE
# ================================================================

def direct_response(
    method,
    state,
    q,
    bathymetry,
    time_value,
):

    t = jnp.asarray(
        [
            time_value
        ],
        dtype=jnp.float32,
    )

    q_batch = jnp.asarray(
        q[
            None,
            ...
        ]
    )

    b_batch = jnp.asarray(
        bathymetry[
            None,
            ...
        ]
    )

    with_bathymetry = (
        method._model_apply(
            state.params,
            q_batch,
            t,
            b_batch,
        )
    )

    flat_bed = (
        method._model_apply(
            state.params,
            q_batch,
            t,
            jnp.zeros_like(
                b_batch
            ),
        )
    )

    response = (
        with_bathymetry
        -
        flat_bed
    )

    return np.asarray(
        response[
            0
        ],
        dtype=np.float32,
    )


# ================================================================
# EXPECTED SWE BED RESPONSE
# ================================================================

def expected_response(
    q,
    bathymetry,
    args,
):

    (
        source_x,
        source_y,
    ) = well_balanced_bed_source(
        jnp.asarray(
            q[
                None,
                ...
            ]
        ),
        jnp.asarray(
            bathymetry[
                None,
                ...
            ]
        ),
        dx=args.dx,
        dy=args.dy,
        g=args.gravity,
    )

    source_x = np.asarray(
        source_x[
            0
        ],
        dtype=np.float32,
    )

    source_y = np.asarray(
        source_y[
            0
        ],
        dtype=np.float32,
    )

    expected = np.zeros(
        (
            source_x.shape[
                0
            ],
            source_x.shape[
                1
            ],
            2,
        ),
        dtype=np.float32,
    )

    expected[
        ...,
        0
    ] = -source_x

    expected[
        ...,
        1
    ] = -source_y

    return expected


# ================================================================
# PLOT 3
# DIRECT BED RESPONSE
# ================================================================

def plot_direct_response(
    data,
    pi_method,
    pi_state,
    bed_method,
    bed_state,
    case_name,
    time_index,
    args,
    output_dir,
):

    case = data[
        "cases"
    ][
        case_name
    ]

    q = case[
        "q"
    ][
        time_index
    ]

    bathymetry = case[
        "bathymetry"
    ]

    time_value = float(
        data[
            "time"
        ][
            time_index
        ]
    )

    expected = expected_response(
        q,
        bathymetry,
        args,
    )

    pi_response = direct_response(
        pi_method,
        pi_state,
        q,
        bathymetry,
        time_value,
    )[
        1:-1,
        1:-1,
        1:3,
    ]

    bed_response = direct_response(
        bed_method,
        bed_state,
        q,
        bathymetry,
        time_value,
    )[
        1:-1,
        1:-1,
        1:3,
    ]

    x = data[
        "x"
    ][
        1:-1
    ]

    y = data[
        "y"
    ][
        1:-1
    ]

    X, Y = make_grid(
        x,
        y,
    )

    bathymetry_interior = (
        bathymetry[
            1:-1,
            1:-1,
            0,
        ]
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            14,
            9,
        ),
        constrained_layout=True,
    )

    column_names = [
        "SWE expected",
        "Original PI-CFO",
        "Bed-PI-CFO",
    ]

    for row in range(
        2
    ):

        fields = [
            expected[
                ...,
                row
            ],
            pi_response[
                ...,
                row
            ],
            bed_response[
                ...,
                row
            ],
        ]

        maximum = max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            )
            for field
            in fields
        )

        for column in range(
            3
        ):

            ax = axes[
                row,
                column,
            ]

            mesh = ax.pcolormesh(
                X,
                Y,
                fields[
                    column
                ],
                shading="auto",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )

            add_contours(
                ax,
                X,
                Y,
                bathymetry_interior,
            )

            fig.colorbar(
                mesh,
                ax=ax,
            )

            if row == 0:

                variable_name = (
                    "delta(hu)_t"
                )

            else:

                variable_name = (
                    "delta(hv)_t"
                )

            ax.set_title(
                f"{column_names[column]}\n"
                f"{variable_name}"
            )

            ax.set_xlabel(
                "x"
            )

            ax.set_ylabel(
                "y"
            )

            ax.set_aspect(
                "equal"
            )

    fig.suptitle(
        "Direct condition test: "
        "same q and t, change only bathymetry"
    )

    fig.savefig(
        output_dir
        /
        "03_direct_bed_response_comparison.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        expected,
        pi_response,
        bed_response,
    )


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    rows,
    path,
):

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    dataset_path = resolve_path(
        args.dataset
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # CHECK PATHS EARLY
    # ============================================================

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset does not exist:\n"
            f"{dataset_path}"
        )

    pi_ckpt_dir = resolve_path(
        args.pi_ckpt_dir
    )

    bed_ckpt_dir = resolve_path(
        args.bed_ckpt_dir
    )

    if not pi_ckpt_dir.exists():

        raise FileNotFoundError(
            f"Original PI checkpoint directory "
            f"does not exist:\n"
            f"{pi_ckpt_dir}"
        )

    if not bed_ckpt_dir.exists():

        raise FileNotFoundError(
            f"Bed-PI checkpoint directory "
            f"does not exist:\n"
            f"{bed_ckpt_dir}"
        )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print(
        "\nLoading controlled Gaussian dataset..."
    )

    data = load_dataset(
        dataset_path
    )

    cases = data[
        "cases"
    ]

    time = data[
        "time"
    ]

    time_index = nearest_time(
        time,
        args.time,
    )

    actual_time = float(
        time[
            time_index
        ]
    )

    input_shape = tuple(
        cases[
            "flat"
        ][
            "q"
        ].shape[
            1:
        ]
    )

    condition_shape = tuple(
        cases[
            "flat"
        ][
            "bathymetry"
        ].shape
    )

    print(
        "\n========================================"
    )

    print(
        "COUNTERFACTUAL COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        f"lambda_bed: "
        f"{args.bed_lambda}"
    )

    print(
        f"Selected time: "
        f"{actual_time:.3f}"
    )

    print(
        f"Dataset: "
        f"{dataset_path}"
    )

    print(
        f"Original PI checkpoint: "
        f"{pi_ckpt_dir}"
    )

    print(
        f"Bed-PI checkpoint: "
        f"{bed_ckpt_dir}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # ORIGINAL PI-CFO
    # ============================================================

    print(
        "\nRestoring original PI-CFO..."
    )

    pi_method = build_original_pi(
        input_shape,
        condition_shape,
        args,
    )

    pi_state = restore_state(
        pi_method,
        pi_ckpt_dir,
        "bathy_pi_cfo",
    )

    # ============================================================
    # BED PI-CFO
    # ============================================================

    print(
        "Restoring Bed-PI-CFO..."
    )

    bed_method = build_bed_pi(
        input_shape,
        condition_shape,
        args,
    )

    bed_state = restore_state(
        bed_method,
        bed_ckpt_dir,
        "bathy_bed_pi",
    )

    # ============================================================
    # ROLLOUTS
    # ============================================================

    print(
        "\nOriginal PI-CFO rollouts:"
    )

    pi_prediction = run_cases(
        pi_method,
        pi_state,
        cases,
    )

    print(
        "\nBed-PI-CFO rollouts:"
    )

    bed_prediction = run_cases(
        bed_method,
        bed_state,
        cases,
    )

    # ============================================================
    # FIGURE 1
    # ============================================================

    print(
        "\nCreating terrain-induced "
        "speed comparison..."
    )

    plot_speed_effect(
        data,
        pi_prediction,
        bed_prediction,
        time_index,
        output_dir,
    )

    # ============================================================
    # FIGURE 2
    # ============================================================

    print(
        "Creating terrain-effect "
        "error-vs-time comparison..."
    )

    plot_error_vs_time(
        data,
        pi_prediction,
        bed_prediction,
        output_dir,
    )

    # ============================================================
    # FIGURE 3
    # ============================================================

    print(
        "Creating direct bed-response "
        "comparison..."
    )

    (
        expected_direct,
        pi_direct,
        bed_direct,
    ) = plot_direct_response(
        data,
        pi_method,
        pi_state,
        bed_method,
        bed_state,
        args.direct_case,
        time_index,
        args,
        output_dir,
    )

    # ============================================================
    # FULL-ROLLOUT METRICS
    # ============================================================

    rows = []

    true_flat = cases[
        "flat"
    ][
        "q"
    ]

    for case_name in [
        "hill_left",
        "hill_center",
        "hill_right",
    ]:

        true_hill = cases[
            case_name
        ][
            "q"
        ]

        # --------------------------------------------------------
        # Terrain effect in complete state
        # --------------------------------------------------------

        true_delta_q = (
            true_hill
            -
            true_flat
        )

        pi_delta_q = (
            pi_prediction[
                case_name
            ]
            -
            pi_prediction[
                "flat"
            ]
        )

        bed_delta_q = (
            bed_prediction[
                case_name
            ]
            -
            bed_prediction[
                "flat"
            ]
        )

        # --------------------------------------------------------
        # Terrain-induced velocity at selected time
        # --------------------------------------------------------

        true_flat_velocity = velocity(
            true_flat[
                time_index
            ]
        )

        true_hill_velocity = velocity(
            true_hill[
                time_index
            ]
        )

        pi_flat_velocity = velocity(
            pi_prediction[
                "flat"
            ][
                time_index
            ]
        )

        pi_hill_velocity = velocity(
            pi_prediction[
                case_name
            ][
                time_index
            ]
        )

        bed_flat_velocity = velocity(
            bed_prediction[
                "flat"
            ][
                time_index
            ]
        )

        bed_hill_velocity = velocity(
            bed_prediction[
                case_name
            ][
                time_index
            ]
        )

        true_du = (
            true_hill_velocity[
                0
            ]
            -
            true_flat_velocity[
                0
            ]
        )

        true_dv = (
            true_hill_velocity[
                1
            ]
            -
            true_flat_velocity[
                1
            ]
        )

        pi_du = (
            pi_hill_velocity[
                0
            ]
            -
            pi_flat_velocity[
                0
            ]
        )

        pi_dv = (
            pi_hill_velocity[
                1
            ]
            -
            pi_flat_velocity[
                1
            ]
        )

        bed_du = (
            bed_hill_velocity[
                0
            ]
            -
            bed_flat_velocity[
                0
            ]
        )

        bed_dv = (
            bed_hill_velocity[
                1
            ]
            -
            bed_flat_velocity[
                1
            ]
        )

        true_velocity_effect = np.stack(
            [
                true_du,
                true_dv,
            ],
            axis=-1,
        )

        pi_velocity_effect = np.stack(
            [
                pi_du,
                pi_dv,
            ],
            axis=-1,
        )

        bed_velocity_effect = np.stack(
            [
                bed_du,
                bed_dv,
            ],
            axis=-1,
        )

        row = {
            "case":
                case_name,

            "pi_delta_q_error":
                relative_error(
                    true_delta_q,
                    pi_delta_q,
                ),

            "bed_delta_q_error":
                relative_error(
                    true_delta_q,
                    bed_delta_q,
                ),

            "pi_velocity_cosine":
                vector_cosine(
                    true_du,
                    true_dv,
                    pi_du,
                    pi_dv,
                ),

            "bed_velocity_cosine":
                vector_cosine(
                    true_du,
                    true_dv,
                    bed_du,
                    bed_dv,
                ),

            "pi_velocity_magnitude_ratio":
                norm_ratio(
                    true_velocity_effect,
                    pi_velocity_effect,
                ),

            "bed_velocity_magnitude_ratio":
                norm_ratio(
                    true_velocity_effect,
                    bed_velocity_effect,
                ),
        }

        rows.append(
            row
        )

    # ============================================================
    # SAVE FULL-ROLLOUT TABLE
    # ============================================================

    csv_path = (
        output_dir
        /
        "counterfactual_metrics.csv"
    )

    save_csv(
        rows,
        csv_path,
    )

    # ============================================================
    # DIRECT RESPONSE METRICS
    # ============================================================

    pi_direct_ratio = norm_ratio(
        expected_direct,
        pi_direct,
    )

    bed_direct_ratio = norm_ratio(
        expected_direct,
        bed_direct,
    )

    pi_direct_cosine = vector_cosine(
        expected_direct[
            ...,
            0
        ],
        expected_direct[
            ...,
            1
        ],
        pi_direct[
            ...,
            0
        ],
        pi_direct[
            ...,
            1
        ],
    )

    bed_direct_cosine = vector_cosine(
        expected_direct[
            ...,
            0
        ],
        expected_direct[
            ...,
            1
        ],
        bed_direct[
            ...,
            0
        ],
        bed_direct[
            ...,
            1
        ],
    )

    # ============================================================
    # PRINT TABLE
    # ============================================================

    print(
        "\n"
        +
        "=" * 112
    )

    print(
        "FULL TERRAIN-RESPONSE COMPARISON"
    )

    print(
        "=" * 112
    )

    print(
        f"{'Case':<14}"
        f"{'PI dq err':>13}"
        f"{'Bed dq err':>14}"
        f"{'PI cosine':>12}"
        f"{'Bed cosine':>13}"
        f"{'PI magnitude':>15}"
        f"{'Bed magnitude':>16}"
    )

    print(
        "-" * 112
    )

    for row in rows:

        print(
            f"{row['case']:<14}"
            f"{row['pi_delta_q_error']:>13.5f}"
            f"{row['bed_delta_q_error']:>14.5f}"
            f"{row['pi_velocity_cosine']:>12.4f}"
            f"{row['bed_velocity_cosine']:>13.4f}"
            f"{row['pi_velocity_magnitude_ratio']:>15.4f}"
            f"{row['bed_velocity_magnitude_ratio']:>16.4f}"
        )

    print(
        "=" * 112
    )

    print(
        "\nDIRECT INSTANTANEOUS BED RESPONSE"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Original PI magnitude ratio: "
        f"{pi_direct_ratio:.4f}"
    )

    print(
        "Bed-PI magnitude ratio:      "
        f"{bed_direct_ratio:.4f}"
    )

    print(
        "Original PI cosine:          "
        f"{pi_direct_cosine:.4f}"
    )

    print(
        "Bed-PI cosine:               "
        f"{bed_direct_cosine:.4f}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Magnitude ratio = 1.0 is ideal."
    )

    print(
        "Magnitude ratio > 1.0 means "
        "terrain effect is too strong."
    )

    print(
        "Magnitude ratio < 1.0 means "
        "terrain effect is too weak."
    )

    print(
        "Cosine closer to 1.0 means better "
        "direction/spatial agreement."
    )

    print(
        f"\nSelected time: "
        f"{actual_time:.3f}"
    )

    print(
        f"\nSaved results to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":

    main()