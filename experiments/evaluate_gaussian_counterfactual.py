"""
Controlled test:

    Did CFO actually learn the effect of Gaussian bathymetry?

The experiment compares:

    flat bed
    Gaussian hill left
    Gaussian hill center
    Gaussian hill right

with exactly the same initial free surface.

Outputs
-------
1. Terrain-induced velocity-speed difference maps
2. Terrain-induced velocity-vector change
3. Terrain-effect prediction error versus time
4. Direct condition sensitivity vs the SWE bed-slope term
5. Quantitative metrics CSV

Run:

    conda activate cfo

    python experiments/evaluate_gaussian_counterfactual.py
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# QUIET LOGGING
# ================================================================

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "3",
)

os.environ.setdefault(
    "ABSL_MIN_LOG_LEVEL",
    "3",
)


# ================================================================
# PROJECT ROOT
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


# ================================================================
# PROJECT IMPORTS
# ================================================================

from bathy_pi_cfo import (
    BathymetryPhysicsInformedCFO,
)

from cfo import (
    ContinuousFlowOperator,
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

def parse_args() -> argparse.Namespace:

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
        "--pi-prefix",
        type=str,
        default="bathy_pi_cfo",
    )

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.03,
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
        "--gravity",
        type=float,
        default=1.0,
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
        "--gamma",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--spline-type",
        type=str,
        default="quintic",
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--quiver-skip",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "gaussian_counterfactual"
        ),
    )

    return parser.parse_args()


# ================================================================
# PATH
# ================================================================

def resolve_path(
    value: str,
) -> Path:

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
# DATA LOADER
# ================================================================

def load_counterfactual_dataset(
    path: Path,
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

            group = h5[
                "cases"
            ][
                case_name
            ]

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

                "xc":
                    float(
                        group.attrs[
                            "xc"
                        ]
                    ),

                "yc":
                    float(
                        group.attrs[
                            "yc"
                        ]
                    ),
            }

    return output


# ================================================================
# MODEL BUILDERS
# ================================================================

def build_cfo(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        gamma=args.gamma,
        spline_type=args.spline_type,
        use_condition=True,
        condition_shape=(
            condition_shape
        ),
    )


def build_pi_cfo(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return (
        BathymetryPhysicsInformedCFO(
            model=model,
            input_shape=input_shape,
            condition_shape=(
                condition_shape
            ),
            gamma=args.gamma,
            spline_type=(
                args.spline_type
            ),
            lambda_pde=(
                args.lambda_pde
            ),
            dx=args.dx,
            dy=args.dy,
            gravity=args.gravity,
        )
    )


# ================================================================
# RESTORE STATE
# ================================================================

def restore_state(
    method,
    *,
    ckpt_dir,
    prefix,
    seed,
):

    target_state = (
        init_cfo_train_state(
            method,
            seed=seed,
            learning_rate=1e-4,
            beta1=0.9,
            beta2=0.99,
        )
    )

    return load_train_state(
        target_state,
        str(
            ckpt_dir
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )


# ================================================================
# VELOCITY
# ================================================================

def velocity(
    q: np.ndarray,
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
# NEAREST TIME
# ================================================================

def nearest_time_index(
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
# RELATIVE EFFECT ERROR
# ================================================================

def relative_effect_error(
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
# VECTOR COSINE SIMILARITY
# ================================================================

def vector_cosine_similarity(
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

    if denominator < 1e-12:

        return np.nan

    return float(
        np.dot(
            true_vector,
            pred_vector,
        )
        /
        denominator
    )


# ================================================================
# EFFECT CENTROID
# ================================================================

def effect_centroid(
    effect,
    X,
    Y,
):

    weights = np.abs(
        effect
    )

    total = float(
        np.sum(
            weights
        )
    )

    if total < 1e-12:

        return (
            np.nan,
            np.nan,
        )

    xc = float(
        np.sum(
            weights
            *
            X
        )
        /
        total
    )

    yc = float(
        np.sum(
            weights
            *
            Y
        )
        /
        total
    )

    return (
        xc,
        yc,
    )


# ================================================================
# RUN ONE MODEL ON ALL CASES
# ================================================================

def run_all_cases(
    method,
    state,
    cases,
    *,
    steps_per_segment,
):

    predictions = {}

    for (
        case_name,
        case,
    ) in cases.items():

        print(
            f"  inference: {case_name}"
        )

        q_true = case[
            "q"
        ]

        b = case[
            "bathymetry"
        ]

        pred = method.uniform_inference(
            state,
            q_true[
                0
            ][
                None,
                ...
            ],
            trajectory_points_num=(
                q_true.shape[
                    0
                ]
            ),
            steps_per_segment=(
                steps_per_segment
            ),
            condition=b[
                None,
                ...
            ],
            method="RK4",
        )

        predictions[
            case_name
        ] = np.asarray(
            pred[
                0
            ],
            dtype=np.float32,
        )

    return predictions


# ================================================================
# PLOT HELPERS
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


def add_hill_contours(
    ax,
    X,
    Y,
    b,
):

    bmax = float(
        np.max(
            b
        )
    )

    if bmax <= 0:

        return

    levels = np.linspace(
        0.2
        *
        bmax,
        0.9
        *
        bmax,
        5,
    )

    ax.contour(
        X,
        Y,
        b,
        levels=levels,
        linewidths=0.8,
    )


# ================================================================
# PLOT 1
# TERRAIN-INDUCED SPEED DIFFERENCE
# ================================================================

def plot_terrain_induced_speed(
    *,
    data,
    cfo_pred,
    pi_pred,
    time_index,
    output_dir,
):

    x = data[
        "x"
    ]

    y = data[
        "y"
    ]

    X, Y = make_grid(
        x,
        y,
    )

    cases = data[
        "cases"
    ]

    flat_true_speed = velocity(
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

    flat_cfo_speed = velocity(
        cfo_pred[
            "flat"
        ][
            time_index
        ]
    )[
        2
    ]

    flat_pi_speed = velocity(
        pi_pred[
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

    model_titles = [
        "PyClaw",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    for row, case_name in enumerate(
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

        cfo_speed = velocity(
            cfo_pred[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        pi_speed = velocity(
            pi_pred[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        differences = [
            true_speed
            -
            flat_true_speed,

            cfo_speed
            -
            flat_cfo_speed,

            pi_speed
            -
            flat_pi_speed,
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
            in differences
        )

        b = cases[
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
                differences[
                    column
                ],
                shading="auto",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )

            add_hill_contours(
                ax,
                X,
                Y,
                b,
            )

            fig.colorbar(
                mesh,
                ax=ax,
                label=(
                    "Δ speed = "
                    "|u|hill - |u|flat"
                ),
            )

            ax.set_title(
                f"{case_name} — "
                f"{model_titles[column]}"
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
        "Terrain-induced change in flow speed"
    )

    fig.savefig(
        output_dir
        /
        "01_terrain_induced_speed_change.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 2
# TERRAIN-INDUCED VELOCITY VECTOR DIFFERENCE
# ================================================================

def plot_velocity_difference_vectors(
    *,
    data,
    cfo_pred,
    pi_pred,
    case_name,
    time_index,
    quiver_skip,
    output_dir,
):

    X, Y = make_grid(
        data[
            "x"
        ],
        data[
            "y"
        ],
    )

    cases = data[
        "cases"
    ]

    true_flat = velocity(
        cases[
            "flat"
        ][
            "q"
        ][
            time_index
        ]
    )

    cfo_flat = velocity(
        cfo_pred[
            "flat"
        ][
            time_index
        ]
    )

    pi_flat = velocity(
        pi_pred[
            "flat"
        ][
            time_index
        ]
    )

    true_hill = velocity(
        cases[
            case_name
        ][
            "q"
        ][
            time_index
        ]
    )

    cfo_hill = velocity(
        cfo_pred[
            case_name
        ][
            time_index
        ]
    )

    pi_hill = velocity(
        pi_pred[
            case_name
        ][
            time_index
        ]
    )

    velocity_differences = [
        (
            true_hill[
                0
            ]
            -
            true_flat[
                0
            ],
            true_hill[
                1
            ]
            -
            true_flat[
                1
            ],
        ),

        (
            cfo_hill[
                0
            ]
            -
            cfo_flat[
                0
            ],
            cfo_hill[
                1
            ]
            -
            cfo_flat[
                1
            ],
        ),

        (
            pi_hill[
                0
            ]
            -
            pi_flat[
                0
            ],
            pi_hill[
                1
            ]
            -
            pi_flat[
                1
            ],
        ),
    ]

    magnitudes = [
        np.sqrt(
            du**2
            +
            dv**2
        )
        for (
            du,
            dv,
        )
        in velocity_differences
    ]

    maximum = max(
        float(
            np.max(
                magnitude
            )
        )
        for magnitude
        in magnitudes
    )

    b = cases[
        case_name
    ][
        "bathymetry"
    ][
        ...,
        0
    ]

    titles = [
        "PyClaw",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5,
        ),
        constrained_layout=True,
    )

    for (
        ax,
        title,
        difference,
        magnitude,
    ) in zip(
        axes,
        titles,
        velocity_differences,
        magnitudes,
    ):

        (
            du,
            dv,
        ) = difference

        mesh = ax.pcolormesh(
            X,
            Y,
            magnitude,
            shading="auto",
            vmin=0.0,
            vmax=maximum,
        )

        add_hill_contours(
            ax,
            X,
            Y,
            b,
        )

        ax.quiver(
            X[
                ::quiver_skip,
                ::quiver_skip,
            ],
            Y[
                ::quiver_skip,
                ::quiver_skip,
            ],
            du[
                ::quiver_skip,
                ::quiver_skip,
            ],
            dv[
                ::quiver_skip,
                ::quiver_skip,
            ],
        )

        fig.colorbar(
            mesh,
            ax=ax,
            label="|Δ velocity|",
        )

        ax.set_title(
            title
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
        f"Velocity change caused specifically "
        f"by {case_name}"
    )

    fig.savefig(
        output_dir
        /
        (
            "02_terrain_induced_velocity_"
            f"{case_name}.png"
        ),
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# TERRAIN EFFECT ERROR THROUGH TIME
# ================================================================

def terrain_effect_error_series(
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

    T = truth_delta.shape[
        0
    ]

    errors = np.zeros(
        T,
        dtype=np.float64,
    )

    for i in range(
        T
    ):

        errors[
            i
        ] = relative_effect_error(
            truth_delta[
                i
            ],
            pred_delta[
                i
            ],
        )

    return errors


# ================================================================
# PLOT 3
# TERRAIN EFFECT ERROR VS TIME
# ================================================================

def plot_effect_error_vs_time(
    *,
    data,
    cfo_pred,
    pi_pred,
    output_dir,
):

    time = data[
        "time"
    ]

    cases = data[
        "cases"
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

        cfo_error = (
            terrain_effect_error_series(
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
                cfo_pred[
                    "flat"
                ],
                cfo_pred[
                    case_name
                ],
            )
        )

        pi_error = (
            terrain_effect_error_series(
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
                pi_pred[
                    "flat"
                ],
                pi_pred[
                    case_name
                ],
            )
        )

        ax.plot(
            time,
            cfo_error,
            label="Bathy-CFO",
        )

        ax.plot(
            time,
            pi_error,
            label="Bathy-PI-CFO",
        )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Relative error in Δq"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

    fig.suptitle(
        "How accurately does the model reproduce "
        "the change caused by the hill?"
    )

    fig.savefig(
        output_dir
        /
        "03_terrain_effect_error_vs_time.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# DIRECT MODEL VECTOR FIELD
# ================================================================

def model_vector_field(
    method,
    state,
    q,
    time_value,
    bathymetry,
):

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

    t_batch = jnp.asarray(
        [
            time_value
        ],
        dtype=jnp.float32,
    )

    result = method._model_apply(
        state.params,
        q_batch,
        t_batch,
        b_batch,
    )

    return np.asarray(
        result[
            0
        ],
        dtype=np.float32,
    )


# ================================================================
# DIRECT BED-SOURCE EXPECTATION
# ================================================================

def expected_bed_condition_effect(
    q,
    b,
    *,
    dx,
    dy,
    gravity,
):

    q_batch = jnp.asarray(
        q[
            None,
            ...
        ]
    )

    b_batch = jnp.asarray(
        b[
            None,
            ...
        ]
    )

    (
        source_x,
        source_y,
    ) = well_balanced_bed_source(
        q_batch,
        b_batch,
        dx=dx,
        dy=dy,
        g=gravity,
    )

    source_x = np.asarray(
        source_x[
            0
        ]
    )

    source_y = np.asarray(
        source_y[
            0
        ]
    )

    H = source_x.shape[
        0
    ]

    W = source_x.shape[
        1
    ]

    expected = np.zeros(
        (
            H,
            W,
            3,
        ),
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Residual convention:
    #
    # q_t + flux divergence + source = 0
    #
    # Therefore changing b from zero to hill contributes:
    #
    # delta q_t = -source
    # ------------------------------------------------------------

    expected[
        ...,
        1
    ] = -source_x

    expected[
        ...,
        2
    ] = -source_y

    return expected


# ================================================================
# DIRECT CONDITION DIAGNOSTIC
# ================================================================

def direct_condition_diagnostic(
    *,
    method,
    state,
    q,
    bathymetry,
    time_value,
):

    zero_bathymetry = np.zeros_like(
        bathymetry
    )

    field_with_hill = (
        model_vector_field(
            method,
            state,
            q,
            time_value,
            bathymetry,
        )
    )

    field_flat = (
        model_vector_field(
            method,
            state,
            q,
            time_value,
            zero_bathymetry,
        )
    )

    return (
        field_with_hill
        -
        field_flat
    )


# ================================================================
# PLOT 4
# DIRECT CONDITION RESPONSE
# ================================================================

def plot_direct_condition_response(
    *,
    data,
    cfo_method,
    cfo_state,
    pi_method,
    pi_state,
    case_name,
    time_index,
    dx,
    dy,
    gravity,
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

    b = case[
        "bathymetry"
    ]

    time_value = float(
        data[
            "time"
        ][
            time_index
        ]
    )

    expected = (
        expected_bed_condition_effect(
            q,
            b,
            dx=dx,
            dy=dy,
            gravity=gravity,
        )
    )

    cfo_delta = (
        direct_condition_diagnostic(
            method=cfo_method,
            state=cfo_state,
            q=q,
            bathymetry=b,
            time_value=time_value,
        )
    )[
        1:-1,
        1:-1,
        :
    ]

    pi_delta = (
        direct_condition_diagnostic(
            method=pi_method,
            state=pi_state,
            q=q,
            bathymetry=b,
            time_value=time_value,
        )
    )[
        1:-1,
        1:-1,
        :
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

    b_interior = b[
        1:-1,
        1:-1,
        0,
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            14,
            9,
        ),
        constrained_layout=True,
    )

    titles = [
        "SWE expected",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    fields_x = [
        expected[
            ...,
            1
        ],
        cfo_delta[
            ...,
            1
        ],
        pi_delta[
            ...,
            1
        ],
    ]

    fields_y = [
        expected[
            ...,
            2
        ],
        cfo_delta[
            ...,
            2
        ],
        pi_delta[
            ...,
            2
        ],
    ]

    for row, fields in enumerate(
        [
            fields_x,
            fields_y,
        ]
    ):

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

            add_hill_contours(
                ax,
                X,
                Y,
                b_interior,
            )

            fig.colorbar(
                mesh,
                ax=ax,
            )

            if row == 0:

                variable = (
                    "Δ(hu)_t"
                )

            else:

                variable = (
                    "Δ(hv)_t"
                )

            ax.set_title(
                f"{titles[column]} "
                f"{variable}"
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
        "Direct bathymetry-condition test: "
        "does changing only b change q_t correctly?"
    )

    fig.savefig(
        output_dir
        /
        "04_direct_bed_source_condition_test.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        expected,
        cfo_delta,
        pi_delta,
    )


# ================================================================
# DIRECT RESPONSE METRICS
# ================================================================

def direct_response_metrics(
    expected,
    prediction,
):

    expected_momentum = expected[
        ...,
        1:3
    ]

    predicted_momentum = prediction[
        ...,
        1:3
    ]

    relative_error = (
        relative_effect_error(
            expected_momentum,
            predicted_momentum,
        )
    )

    cosine = (
        vector_cosine_similarity(
            expected[
                ...,
                1
            ],
            expected[
                ...,
                2
            ],
            prediction[
                ...,
                1
            ],
            prediction[
                ...,
                2
            ],
        )
    )

    # Continuity equation has no direct bed source.
    continuity_leakage = float(
        np.sqrt(
            np.mean(
                prediction[
                    ...,
                    0
                ]
                **
                2
            )
        )
    )

    return {
        "relative_momentum_error":
            relative_error,

        "momentum_cosine_similarity":
            cosine,

        "continuity_leakage_rms":
            continuity_leakage,
    }


# ================================================================
# SAVE TABLE
# ================================================================

def save_metrics_csv(
    rows,
    output_path,
):

    if not rows:

        return

    fieldnames = list(
        rows[
            0
        ].keys()
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
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

    cfo_ckpt_dir = resolve_path(
        args.cfo_ckpt_dir
    )

    pi_ckpt_dir = resolve_path(
        args.pi_ckpt_dir
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # LOAD CONTROL DATA
    # ============================================================

    print(
        "\nLoading controlled dataset..."
    )

    data = (
        load_counterfactual_dataset(
            dataset_path
        )
    )

    cases = data[
        "cases"
    ]

    time = data[
        "time"
    ]

    time_index = (
        nearest_time_index(
            time,
            args.time,
        )
    )

    actual_time = float(
        time[
            time_index
        ]
    )

    print(
        f"Selected time: "
        f"{actual_time:.4f}"
    )

    input_shape = (
        cases[
            "flat"
        ][
            "q"
        ].shape[
            1:
        ]
    )

    condition_shape = (
        cases[
            "flat"
        ][
            "bathymetry"
        ].shape
    )

    # ============================================================
    # RESTORE CFO
    # ============================================================

    print(
        "\nRestoring Bathy-CFO..."
    )

    cfo_method = build_cfo(
        input_shape,
        condition_shape,
        args,
    )

    cfo_state = restore_state(
        cfo_method,
        ckpt_dir=cfo_ckpt_dir,
        prefix=args.cfo_prefix,
        seed=args.seed,
    )

    # ============================================================
    # RESTORE PI-CFO
    # ============================================================

    print(
        "Restoring Bathy-PI-CFO..."
    )

    pi_method = build_pi_cfo(
        input_shape,
        condition_shape,
        args,
    )

    pi_state = restore_state(
        pi_method,
        ckpt_dir=pi_ckpt_dir,
        prefix=args.pi_prefix,
        seed=args.seed,
    )

    # ============================================================
    # ROLLOUT ALL FOUR CONDITIONS
    # ============================================================

    print(
        "\nRunning Bathy-CFO "
        "counterfactual rollouts..."
    )

    cfo_pred = run_all_cases(
        cfo_method,
        cfo_state,
        cases,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    print(
        "\nRunning Bathy-PI-CFO "
        "counterfactual rollouts..."
    )

    pi_pred = run_all_cases(
        pi_method,
        pi_state,
        cases,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    # ============================================================
    # PLOT 1
    # ============================================================

    plot_terrain_induced_speed(
        data=data,
        cfo_pred=cfo_pred,
        pi_pred=pi_pred,
        time_index=time_index,
        output_dir=output_dir,
    )

    # ============================================================
    # PLOT 2
    # ============================================================

    plot_velocity_difference_vectors(
        data=data,
        cfo_pred=cfo_pred,
        pi_pred=pi_pred,
        case_name=(
            args.direct_case
        ),
        time_index=time_index,
        quiver_skip=(
            args.quiver_skip
        ),
        output_dir=output_dir,
    )

    # ============================================================
    # PLOT 3
    # ============================================================

    plot_effect_error_vs_time(
        data=data,
        cfo_pred=cfo_pred,
        pi_pred=pi_pred,
        output_dir=output_dir,
    )

    # ============================================================
    # PLOT 4
    # ============================================================

    (
        expected_direct,
        cfo_direct,
        pi_direct,
    ) = plot_direct_condition_response(
        data=data,
        cfo_method=cfo_method,
        cfo_state=cfo_state,
        pi_method=pi_method,
        pi_state=pi_state,
        case_name=(
            args.direct_case
        ),
        time_index=time_index,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
        output_dir=output_dir,
    )

    # ============================================================
    # QUANTITATIVE TERRAIN EFFECT METRICS
    # ============================================================

    metric_rows = []

    X, Y = make_grid(
        data[
            "x"
        ],
        data[
            "y"
        ],
    )

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
        # Full trajectory effect:
        #
        # delta q = q_hill - q_flat
        # --------------------------------------------------------

        true_delta_q = (
            true_hill
            -
            true_flat
        )

        cfo_delta_q = (
            cfo_pred[
                case_name
            ]
            -
            cfo_pred[
                "flat"
            ]
        )

        pi_delta_q = (
            pi_pred[
                case_name
            ]
            -
            pi_pred[
                "flat"
            ]
        )

        full_cfo_error = (
            relative_effect_error(
                true_delta_q,
                cfo_delta_q,
            )
        )

        full_pi_error = (
            relative_effect_error(
                true_delta_q,
                pi_delta_q,
            )
        )

        # --------------------------------------------------------
        # Velocity effect at selected time
        # --------------------------------------------------------

        true_hill_vel = velocity(
            true_hill[
                time_index
            ]
        )

        true_flat_vel = velocity(
            true_flat[
                time_index
            ]
        )

        cfo_hill_vel = velocity(
            cfo_pred[
                case_name
            ][
                time_index
            ]
        )

        cfo_flat_vel = velocity(
            cfo_pred[
                "flat"
            ][
                time_index
            ]
        )

        pi_hill_vel = velocity(
            pi_pred[
                case_name
            ][
                time_index
            ]
        )

        pi_flat_vel = velocity(
            pi_pred[
                "flat"
            ][
                time_index
            ]
        )

        true_du = (
            true_hill_vel[
                0
            ]
            -
            true_flat_vel[
                0
            ]
        )

        true_dv = (
            true_hill_vel[
                1
            ]
            -
            true_flat_vel[
                1
            ]
        )

        cfo_du = (
            cfo_hill_vel[
                0
            ]
            -
            cfo_flat_vel[
                0
            ]
        )

        cfo_dv = (
            cfo_hill_vel[
                1
            ]
            -
            cfo_flat_vel[
                1
            ]
        )

        pi_du = (
            pi_hill_vel[
                0
            ]
            -
            pi_flat_vel[
                0
            ]
        )

        pi_dv = (
            pi_hill_vel[
                1
            ]
            -
            pi_flat_vel[
                1
            ]
        )

        cfo_cosine = (
            vector_cosine_similarity(
                true_du,
                true_dv,
                cfo_du,
                cfo_dv,
            )
        )

        pi_cosine = (
            vector_cosine_similarity(
                true_du,
                true_dv,
                pi_du,
                pi_dv,
            )
        )

        # --------------------------------------------------------
        # Effect localization
        # --------------------------------------------------------

        true_speed_delta = (
            true_hill_vel[
                2
            ]
            -
            true_flat_vel[
                2
            ]
        )

        cfo_speed_delta = (
            cfo_hill_vel[
                2
            ]
            -
            cfo_flat_vel[
                2
            ]
        )

        pi_speed_delta = (
            pi_hill_vel[
                2
            ]
            -
            pi_flat_vel[
                2
            ]
        )

        (
            true_xc,
            true_yc,
        ) = effect_centroid(
            true_speed_delta,
            X,
            Y,
        )

        (
            cfo_xc,
            cfo_yc,
        ) = effect_centroid(
            cfo_speed_delta,
            X,
            Y,
        )

        (
            pi_xc,
            pi_yc,
        ) = effect_centroid(
            pi_speed_delta,
            X,
            Y,
        )

        metric_rows.append(
            {
                "case":
                    case_name,

                "full_delta_q_error_cfo":
                    full_cfo_error,

                "full_delta_q_error_pi":
                    full_pi_error,

                "velocity_effect_cosine_cfo":
                    cfo_cosine,

                "velocity_effect_cosine_pi":
                    pi_cosine,

                "true_effect_centroid_x":
                    true_xc,

                "cfo_effect_centroid_x":
                    cfo_xc,

                "pi_effect_centroid_x":
                    pi_xc,

                "true_effect_centroid_y":
                    true_yc,

                "cfo_effect_centroid_y":
                    cfo_yc,

                "pi_effect_centroid_y":
                    pi_yc,
            }
        )

    save_metrics_csv(
        metric_rows,
        output_dir
        /
        "terrain_effect_metrics.csv",
    )

    # ============================================================
    # DIRECT BED RESPONSE METRICS
    # ============================================================

    cfo_direct_metrics = (
        direct_response_metrics(
            expected_direct,
            cfo_direct,
        )
    )

    pi_direct_metrics = (
        direct_response_metrics(
            expected_direct,
            pi_direct,
        )
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "DID THE MODEL LEARN THE GAUSSIAN-HILL EFFECT?"
    )

    print(
        "=" * 90
    )

    print(
        "\nFull-trajectory terrain-effect metrics"
    )

    print(
        "-" * 90
    )

    print(
        f"{'Case':<15}"
        f"{'Δq err CFO':>15}"
        f"{'Δq err PI':>15}"
        f"{'Vel cosine CFO':>18}"
        f"{'Vel cosine PI':>18}"
    )

    for row in metric_rows:

        print(
            f"{row['case']:<15}"
            f"{row['full_delta_q_error_cfo']:>15.6f}"
            f"{row['full_delta_q_error_pi']:>15.6f}"
            f"{row['velocity_effect_cosine_cfo']:>18.6f}"
            f"{row['velocity_effect_cosine_pi']:>18.6f}"
        )

    print(
        "\n"
        +
        "=" * 90
    )

    print(
        "DIRECT BED-CONDITION TEST"
    )

    print(
        "=" * 90
    )

    print(
        f"Case: {args.direct_case}"
    )

    print(
        f"Time: {actual_time:.4f}"
    )

    print(
        "\nBathy-CFO:"
    )

    print(
        "  momentum response relative error: "
        f"{cfo_direct_metrics['relative_momentum_error']:.6f}"
    )

    print(
        "  momentum response cosine similarity: "
        f"{cfo_direct_metrics['momentum_cosine_similarity']:.6f}"
    )

    print(
        "  continuity leakage RMS: "
        f"{cfo_direct_metrics['continuity_leakage_rms']:.6e}"
    )

    print(
        "\nBathy-PI-CFO:"
    )

    print(
        "  momentum response relative error: "
        f"{pi_direct_metrics['relative_momentum_error']:.6f}"
    )

    print(
        "  momentum response cosine similarity: "
        f"{pi_direct_metrics['momentum_cosine_similarity']:.6f}"
    )

    print(
        "  continuity leakage RMS: "
        f"{pi_direct_metrics['continuity_leakage_rms']:.6e}"
    )

    print(
        "\nFigures and table saved to:"
    )

    print(
        output_dir
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    main()