"""
Generate the eight main diagnostic / physics plots for the
bathymetry-conditioned CFO project.

Models compared
---------------
1. PyClaw ground truth
2. Bathy-CFO
3. Bathy-PI-CFO

Default PI model
----------------
lambda_PDE = 0.03

Default data
------------
Validation split ("eval").

After lambda_PDE is frozen, run with:

    --split test_id

Outputs
-------
01. Bathymetry + free-surface elevation
02. Reference vs CFO vs PI-CFO state fields
03. Absolute error maps
04. Velocity-vector / quiver plots over terrain
05. Velocity-magnitude maps over terrain
06. Cross-sections through the Gaussian hill
07. SWE residual versus time
    a) own-rollout residual
    b) same-ground-truth vector-field residual
08. Relative mass drift versus time

Run from project root:

    python experiments/plot_bathy_all_eight.py

Example for test set later:

    python experiments/plot_bathy_all_eight.py --split test_id
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys

import jax
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

from cfo import (
    ContinuousFlowOperator,
)

from models.factory import (
    build_model,
)

from train import (
    init_cfo_train_state,
)

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.checkpoints import (
    load_train_state,
)

from utils.physics_swe_bathy import (
    swe_bathy_residual,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Generate all eight bathymetry "
            "diagnostic plot families."
        )
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        choices=[
            "eval",
            "test_id",
        ],
    )

    parser.add_argument(
        "--trajectory",
        type=int,
        default=0,
        help=(
            "Trajectory used for spatial "
            "field visualizations."
        ),
    )

    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs="+",
        default=[
            0.25,
            0.50,
            0.75,
            1.00,
        ],
    )

    parser.add_argument(
        "--free-surface-time",
        type=float,
        default=0.50,
    )

    # ------------------------------------------------------------
    # CFO checkpoint
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # PI-CFO checkpoint
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------

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
        "--gamma",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--spline-type",
        type=str,
        default="quintic",
    )

    # ------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------

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
        "--vector-field-batch-size",
        type=int,
        default=8,
    )

    # ------------------------------------------------------------
    # Quiver
    # ------------------------------------------------------------

    parser.add_argument(
        "--quiver-skip",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--quiver-scale",
        type=float,
        default=2.5,
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "bathy_all_eight_plots"
        ),
    )

    return parser.parse_args()


# ================================================================
# PATH HELPER
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
# BUILD CFO METHOD
# ================================================================

def build_cfo_method(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    method = ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        gamma=float(
            args.gamma
        ),
        spline_type=(
            args.spline_type
        ),
        use_condition=True,
        condition_shape=(
            condition_shape
        ),
    )

    return method


# ================================================================
# BUILD PI-CFO METHOD
# ================================================================

def build_pi_method(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    method = (
        BathymetryPhysicsInformedCFO(
            model=model,
            input_shape=input_shape,
            condition_shape=(
                condition_shape
            ),
            gamma=float(
                args.gamma
            ),
            spline_type=(
                args.spline_type
            ),
            lambda_pde=float(
                args.lambda_pde
            ),
            dx=float(
                args.dx
            ),
            dy=float(
                args.dy
            ),
            gravity=float(
                args.gravity
            ),
        )
    )

    return method


# ================================================================
# RESTORE CHECKPOINT
# ================================================================

def restore_method(
    method,
    *,
    ckpt_dir: Path,
    prefix: str,
    seed: int,
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

    state = load_train_state(
        target_state,
        str(
            ckpt_dir
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )

    return state


# ================================================================
# FIND NEAREST TIME INDEX
# ================================================================

def nearest_time_index(
    time: np.ndarray,
    target: float,
) -> int:

    return int(
        np.argmin(
            np.abs(
                time
                -
                float(
                    target
                )
            )
        )
    )


# ================================================================
# FORMAT TIME FOR FILE NAMES
# ================================================================

def time_label(
    value: float,
) -> str:

    return (
        f"{value:.2f}"
        .replace(
            ".",
            "p",
        )
    )


# ================================================================
# RELATIVE L2
# ================================================================

def relative_l2(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:

    numerator = np.linalg.norm(
        (
            prediction
            -
            truth
        ).reshape(
            truth.shape[
                0
            ],
            -1,
        ),
        axis=1,
    )

    denominator = np.linalg.norm(
        truth.reshape(
            truth.shape[
                0
            ],
            -1,
        ),
        axis=1,
    )

    values = (
        numerator
        /
        np.maximum(
            denominator,
            1e-12,
        )
    )

    return float(
        np.mean(
            values
        )
    )


# ================================================================
# VELOCITY COMPONENTS
# ================================================================

def velocity_components(
    q: np.ndarray,
):

    h = q[
        ...,
        0
    ]

    hu = q[
        ...,
        1
    ]

    hv = q[
        ...,
        2
    ]

    h_safe = np.maximum(
        h,
        1e-6,
    )

    u = (
        hu
        /
        h_safe
    )

    v = (
        hv
        /
        h_safe
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
# FREE SURFACE
# ================================================================

def free_surface(
    q: np.ndarray,
    bathymetry: np.ndarray,
):

    return (
        q[
            ...,
            0
        ]
        +
        bathymetry[
            ...,
            0
        ]
    )


# ================================================================
# TIME DERIVATIVE
# ================================================================

def finite_difference_time_derivative(
    q: np.ndarray,
    time: np.ndarray,
):

    edge_order = (
        2
        if len(
            time
        ) >= 3
        else 1
    )

    q_t = np.gradient(
        q,
        time,
        axis=1,
        edge_order=edge_order,
    )

    return np.asarray(
        q_t,
        dtype=np.float32,
    )


# ================================================================
# SWE RESIDUAL TIME SERIES
# ================================================================

def residual_time_series(
    q: np.ndarray,
    q_t: np.ndarray,
    bathymetry: np.ndarray,
    *,
    dx: float,
    dy: float,
    gravity: float,
) -> np.ndarray:

    B = q.shape[
        0
    ]

    T = q.shape[
        1
    ]

    values = np.zeros(
        T,
        dtype=np.float64,
    )

    for t_index in range(
        T
    ):

        residual = (
            swe_bathy_residual(
                jnp.asarray(
                    q[
                        :,
                        t_index,
                    ]
                ),
                jnp.asarray(
                    q_t[
                        :,
                        t_index,
                    ]
                ),
                jnp.asarray(
                    bathymetry
                ),
                dx=float(
                    dx
                ),
                dy=float(
                    dy
                ),
                g=float(
                    gravity
                ),
            )
        )

        residual = np.asarray(
            residual,
            dtype=np.float64,
        )

        values[
            t_index
        ] = float(
            np.mean(
                residual**2
            )
        )

    return values


# ================================================================
# MODEL VECTOR FIELD ON GROUND TRUTH
# ================================================================

def model_vector_field_on_ground_truth(
    method,
    state,
    q_true: np.ndarray,
    bathymetry: np.ndarray,
    time: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:

    B, T, H, W, C = (
        q_true.shape
    )

    q_flat = q_true.reshape(
        B * T,
        H,
        W,
        C,
    )

    b_flat = np.repeat(
        bathymetry,
        T,
        axis=0,
    )

    t_flat = np.tile(
        time,
        B,
    )

    output = []

    for start in range(
        0,
        B * T,
        batch_size,
    ):

        end = min(
            start
            +
            batch_size,
            B * T,
        )

        q_batch = jnp.asarray(
            q_flat[
                start:end
            ]
        )

        b_batch = jnp.asarray(
            b_flat[
                start:end
            ]
        )

        t_batch = jnp.asarray(
            t_flat[
                start:end
            ]
        )

        pred = method._model_apply(
            state.params,
            q_batch,
            t_batch,
            b_batch,
        )

        output.append(
            np.asarray(
                pred,
                dtype=np.float32,
            )
        )

    q_t = np.concatenate(
        output,
        axis=0,
    )

    return q_t.reshape(
        B,
        T,
        H,
        W,
        C,
    )


# ================================================================
# MASS DRIFT
# ================================================================

def mass_drift_time_series(
    q: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> np.ndarray:

    h = q[
        ...,
        0
    ]

    mass = (
        np.sum(
            h,
            axis=(
                2,
                3,
            ),
        )
        *
        float(
            dx
        )
        *
        float(
            dy
        )
    )

    initial = mass[
        :,
        0:1
    ]

    relative_drift = (
        np.abs(
            mass
            -
            initial
        )
        /
        np.maximum(
            np.abs(
                initial
            ),
            1e-12,
        )
    )

    return np.mean(
        relative_drift,
        axis=0,
    )


# ================================================================
# COMMON X/Y GRID
# ================================================================

def make_xy_grid(
    x: np.ndarray,
    y: np.ndarray,
):

    return np.meshgrid(
        x,
        y,
        indexing="ij",
    )


# ================================================================
# ADD BATHYMETRY CONTOURS
# ================================================================

def add_bathy_contours(
    ax,
    X,
    Y,
    b,
):

    b_min = float(
        np.min(
            b
        )
    )

    b_max = float(
        np.max(
            b
        )
    )

    if (
        b_max
        >
        b_min
    ):

        levels = np.linspace(
            b_min
            +
            0.15
            *
            (
                b_max
                -
                b_min
            ),
            b_max
            *
            0.95,
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
# BATHYMETRY + FREE SURFACE
# ================================================================

def plot_01_bathymetry_free_surface(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    eta_initial = (
        q_true[
            trajectory,
            0,
            ...,
            0,
        ]
        +
        b
    )

    eta_true = (
        q_true[
            trajectory,
            idx,
            ...,
            0,
        ]
        +
        b
    )

    eta_cfo = (
        q_cfo[
            trajectory,
            idx,
            ...,
            0,
        ]
        +
        b
    )

    eta_pi = (
        q_pi[
            trajectory,
            idx,
            ...,
            0,
        ]
        +
        b
    )

    eta_all = np.stack(
        [
            eta_initial,
            eta_true,
            eta_cfo,
            eta_pi,
        ]
    )

    eta_min = float(
        np.min(
            eta_all
        )
    )

    eta_max = float(
        np.max(
            eta_all
        )
    )

    X, Y = make_xy_grid(
        x,
        y,
    )

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(
            21,
            4,
        ),
        constrained_layout=True,
    )

    mesh = axes[
        0
    ].pcolormesh(
        X,
        Y,
        b,
        shading="auto",
    )

    fig.colorbar(
        mesh,
        ax=axes[
            0
        ],
        label="b",
    )

    axes[
        0
    ].set_title(
        "Bathymetry"
    )

    fields = [
        (
            eta_initial,
            "Initial free surface",
        ),
        (
            eta_true,
            f"PyClaw η, t={actual_time:.2f}",
        ),
        (
            eta_cfo,
            f"CFO η, t={actual_time:.2f}",
        ),
        (
            eta_pi,
            f"PI-CFO η, t={actual_time:.2f}",
        ),
    ]

    for ax, (
        field,
        title,
    ) in zip(
        axes[
            1:
        ],
        fields,
    ):

        mesh = ax.pcolormesh(
            X,
            Y,
            field,
            shading="auto",
            vmin=eta_min,
            vmax=eta_max,
        )

        add_bathy_contours(
            ax,
            X,
            Y,
            b,
        )

        fig.colorbar(
            mesh,
            ax=ax,
            label="η = h + b",
        )

        ax.set_title(
            title
        )

    for ax in axes:

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
        "Plot 1 — Bathymetry and free-surface response"
    )

    path = (
        output_dir
        /
        "01_bathymetry_and_free_surface.png"
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# FIELD COLOR LIMITS
# ================================================================

def state_color_limits(
    fields,
    channel: int,
):

    stacked = np.stack(
        fields
    )

    if channel == 0:

        return (
            float(
                np.min(
                    stacked
                )
            ),
            float(
                np.max(
                    stacked
                )
            ),
        )

    max_abs = float(
        np.max(
            np.abs(
                stacked
            )
        )
    )

    return (
        -max_abs,
        max_abs,
    )


# ================================================================
# PLOT 2
# REFERENCE VS CFO VS PI-CFO
# ================================================================

def plot_02_field_comparison(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    X, Y = make_xy_grid(
        x,
        y,
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    variable_names = [
        "h",
        "hu",
        "hv",
    ]

    model_names = [
        "PyClaw",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    model_arrays = [
        q_true,
        q_cfo,
        q_pi,
    ]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(
            13,
            12,
        ),
        constrained_layout=True,
    )

    for channel in range(
        3
    ):

        fields = [
            model[
                trajectory,
                idx,
                ...,
                channel,
            ]
            for model
            in model_arrays
        ]

        (
            vmin,
            vmax,
        ) = state_color_limits(
            fields,
            channel,
        )

        for column in range(
            3
        ):

            ax = axes[
                channel,
                column,
            ]

            field = fields[
                column
            ]

            mesh = ax.pcolormesh(
                X,
                Y,
                field,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )

            add_bathy_contours(
                ax,
                X,
                Y,
                b,
            )

            fig.colorbar(
                mesh,
                ax=ax,
                label=variable_names[
                    channel
                ],
            )

            ax.set_title(
                f"{model_names[column]} "
                f"{variable_names[channel]}"
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
        f"Plot 2 — State comparison at t={actual_time:.2f}"
    )

    path = (
        output_dir
        /
        (
            "02_field_comparison_"
            f"t{time_label(actual_time)}.png"
        )
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 3
# ABSOLUTE ERROR MAPS
# ================================================================

def plot_03_error_maps(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    X, Y = make_xy_grid(
        x,
        y,
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    variable_names = [
        "h",
        "hu",
        "hv",
    ]

    cfo_error = np.abs(
        q_cfo[
            trajectory,
            idx,
        ]
        -
        q_true[
            trajectory,
            idx,
        ]
    )

    pi_error = np.abs(
        q_pi[
            trajectory,
            idx,
        ]
        -
        q_true[
            trajectory,
            idx,
        ]
    )

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(
            10,
            12,
        ),
        constrained_layout=True,
    )

    for channel in range(
        3
    ):

        maximum = float(
            np.max(
                [
                    np.max(
                        cfo_error[
                            ...,
                            channel,
                        ]
                    ),
                    np.max(
                        pi_error[
                            ...,
                            channel,
                        ]
                    ),
                ]
            )
        )

        for column, (
            error,
            label,
        ) in enumerate(
            [
                (
                    cfo_error[
                        ...,
                        channel,
                    ],
                    "CFO",
                ),
                (
                    pi_error[
                        ...,
                        channel,
                    ],
                    "PI-CFO",
                ),
            ]
        ):

            ax = axes[
                channel,
                column,
            ]

            mesh = ax.pcolormesh(
                X,
                Y,
                error,
                shading="auto",
                vmin=0.0,
                vmax=maximum,
            )

            add_bathy_contours(
                ax,
                X,
                Y,
                b,
            )

            fig.colorbar(
                mesh,
                ax=ax,
                label=(
                    f"|Δ{variable_names[channel]}|"
                ),
            )

            ax.set_title(
                f"{label} absolute "
                f"{variable_names[channel]} error"
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
        f"Plot 3 — Absolute errors at t={actual_time:.2f}"
    )

    path = (
        output_dir
        /
        (
            "03_absolute_errors_"
            f"t{time_label(actual_time)}.png"
        )
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 4
# VELOCITY VECTORS OVER TERRAIN
# ================================================================

def plot_04_velocity_vectors(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    skip,
    quiver_scale,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    X, Y = make_xy_grid(
        x,
        y,
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    model_arrays = [
        q_true,
        q_cfo,
        q_pi,
    ]

    model_names = [
        "PyClaw",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    velocities = [
        velocity_components(
            model[
                trajectory,
                idx,
            ]
        )
        for model
        in model_arrays
    ]

    speed_max = max(
        float(
            np.max(
                velocity[
                    2
                ]
            )
        )
        for velocity
        in velocities
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5,
        ),
        constrained_layout=True,
    )

    for ax, (
        model_name,
        velocity,
    ) in zip(
        axes,
        zip(
            model_names,
            velocities,
        ),
    ):

        (
            u,
            v,
            speed,
        ) = velocity

        mesh = ax.pcolormesh(
            X,
            Y,
            speed,
            shading="auto",
            vmin=0.0,
            vmax=speed_max,
        )

        add_bathy_contours(
            ax,
            X,
            Y,
            b,
        )

        ax.quiver(
            X[
                ::skip,
                ::skip,
            ],
            Y[
                ::skip,
                ::skip,
            ],
            u[
                ::skip,
                ::skip,
            ],
            v[
                ::skip,
                ::skip,
            ],
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.003,
        )

        fig.colorbar(
            mesh,
            ax=ax,
            label="velocity magnitude",
        )

        ax.set_title(
            model_name
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
        f"Plot 4 — Flow vectors around terrain at "
        f"t={actual_time:.2f}"
    )

    path = (
        output_dir
        /
        (
            "04_velocity_vectors_"
            f"t{time_label(actual_time)}.png"
        )
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 5
# VELOCITY MAGNITUDE
# ================================================================

def plot_05_velocity_magnitude(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    X, Y = make_xy_grid(
        x,
        y,
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    model_arrays = [
        q_true,
        q_cfo,
        q_pi,
    ]

    model_names = [
        "PyClaw",
        "Bathy-CFO",
        "Bathy-PI-CFO",
    ]

    speeds = [
        velocity_components(
            model[
                trajectory,
                idx,
            ]
        )[
            2
        ]
        for model
        in model_arrays
    ]

    vmax = float(
        np.max(
            np.stack(
                speeds
            )
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            16,
            5,
        ),
        constrained_layout=True,
    )

    for ax, (
        speed,
        model_name,
    ) in zip(
        axes,
        zip(
            speeds,
            model_names,
        ),
    ):

        mesh = ax.pcolormesh(
            X,
            Y,
            speed,
            shading="auto",
            vmin=0.0,
            vmax=vmax,
        )

        add_bathy_contours(
            ax,
            X,
            Y,
            b,
        )

        fig.colorbar(
            mesh,
            ax=ax,
            label="|u|",
        )

        ax.set_title(
            model_name
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
        f"Plot 5 — Velocity magnitude around the hill "
        f"at t={actual_time:.2f}"
    )

    path = (
        output_dir
        /
        (
            "05_velocity_magnitude_"
            f"t{time_label(actual_time)}.png"
        )
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 6
# CROSS-SECTIONS THROUGH HILL CENTER
# ================================================================

def plot_06_cross_sections(
    *,
    x,
    y,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    trajectory,
    target_time,
    output_dir,
):

    idx = nearest_time_index(
        time,
        target_time,
    )

    actual_time = float(
        time[
            idx
        ]
    )

    b = bathymetry[
        trajectory,
        ...,
        0,
    ]

    # ------------------------------------------------------------
    # Gaussian hill center estimated from maximum bathymetry.
    # ------------------------------------------------------------

    hill_index = np.unravel_index(
        np.argmax(
            b
        ),
        b.shape,
    )

    ix = int(
        hill_index[
            0
        ]
    )

    iy = int(
        hill_index[
            1
        ]
    )

    xc = float(
        x[
            ix
        ]
    )

    yc = float(
        y[
            iy
        ]
    )

    variables = [
        "h",
        "hu",
        "hv",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            16,
            9,
        ),
        constrained_layout=True,
    )

    # ============================================================
    # X DIRECTION:
    # y = hill center
    # ============================================================

    for channel in range(
        3
    ):

        ax = axes[
            0,
            channel,
        ]

        ax.plot(
            x,
            q_true[
                trajectory,
                idx,
                :,
                iy,
                channel,
            ],
            label="PyClaw",
        )

        ax.plot(
            x,
            q_cfo[
                trajectory,
                idx,
                :,
                iy,
                channel,
            ],
            label="CFO",
        )

        ax.plot(
            x,
            q_pi[
                trajectory,
                idx,
                :,
                iy,
                channel,
            ],
            label="PI-CFO",
        )

        ax.set_title(
            f"{variables[channel]} along y={yc:.3f}"
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            variables[
                channel
            ]
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        # --------------------------------------------------------
        # Overlay hill profile.
        # --------------------------------------------------------

        terrain_axis = (
            ax.twinx()
        )

        terrain_axis.plot(
            x,
            b[
                :,
                iy,
            ],
            linestyle="--",
            alpha=0.45,
            label="bathymetry",
        )

        terrain_axis.set_ylabel(
            "b"
        )

    # ============================================================
    # Y DIRECTION:
    # x = hill center
    # ============================================================

    for channel in range(
        3
    ):

        ax = axes[
            1,
            channel,
        ]

        ax.plot(
            y,
            q_true[
                trajectory,
                idx,
                ix,
                :,
                channel,
            ],
            label="PyClaw",
        )

        ax.plot(
            y,
            q_cfo[
                trajectory,
                idx,
                ix,
                :,
                channel,
            ],
            label="CFO",
        )

        ax.plot(
            y,
            q_pi[
                trajectory,
                idx,
                ix,
                :,
                channel,
            ],
            label="PI-CFO",
        )

        ax.set_title(
            f"{variables[channel]} along x={xc:.3f}"
        )

        ax.set_xlabel(
            "y"
        )

        ax.set_ylabel(
            variables[
                channel
            ]
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend()

        terrain_axis = (
            ax.twinx()
        )

        terrain_axis.plot(
            y,
            b[
                ix,
                :,
            ],
            linestyle="--",
            alpha=0.45,
            label="bathymetry",
        )

        terrain_axis.set_ylabel(
            "b"
        )

    fig.suptitle(
        "Plot 6 — Cross-sections through Gaussian hill "
        f"at t={actual_time:.2f}"
    )

    path = (
        output_dir
        /
        (
            "06_cross_sections_"
            f"t{time_label(actual_time)}.png"
        )
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT 7A
# OWN-ROLLOUT PDE RESIDUAL
# ================================================================

def plot_07a_rollout_residual(
    *,
    time,
    q_true,
    q_cfo,
    q_pi,
    bathymetry,
    dx,
    dy,
    gravity,
    output_dir,
):

    q_t_true = (
        finite_difference_time_derivative(
            q_true,
            time,
        )
    )

    q_t_cfo = (
        finite_difference_time_derivative(
            q_cfo,
            time,
        )
    )

    q_t_pi = (
        finite_difference_time_derivative(
            q_pi,
            time,
        )
    )

    residual_true = residual_time_series(
        q_true,
        q_t_true,
        bathymetry,
        dx=dx,
        dy=dy,
        gravity=gravity,
    )

    residual_cfo = residual_time_series(
        q_cfo,
        q_t_cfo,
        bathymetry,
        dx=dx,
        dy=dy,
        gravity=gravity,
    )

    residual_pi = residual_time_series(
        q_pi,
        q_t_pi,
        bathymetry,
        dx=dx,
        dy=dy,
        gravity=gravity,
    )

    fig = plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.semilogy(
        time,
        residual_true,
        label="PyClaw FD reference",
    )

    plt.semilogy(
        time,
        residual_cfo,
        label="Bathy-CFO",
    )

    plt.semilogy(
        time,
        residual_pi,
        label="Bathy-PI-CFO",
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Variable-bottom SWE residual MSE"
    )

    plt.title(
        "Plot 7A — SWE residual on each model's own rollout"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        output_dir
        /
        "07a_rollout_swe_residual_vs_time.png"
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        residual_true,
        residual_cfo,
        residual_pi,
    )


# ================================================================
# PLOT 7B
# SAME-GROUND-TRUTH VECTOR FIELD PDE RESIDUAL
# ================================================================

def plot_07b_same_gt_vector_field(
    *,
    time,
    q_true,
    bathymetry,
    cfo_method,
    cfo_state,
    pi_method,
    pi_state,
    dx,
    dy,
    gravity,
    batch_size,
    output_dir,
):

    print(
        "\nEvaluating CFO vector field "
        "on ground-truth states..."
    )

    q_t_cfo = (
        model_vector_field_on_ground_truth(
            cfo_method,
            cfo_state,
            q_true,
            bathymetry,
            time,
            batch_size=batch_size,
        )
    )

    print(
        "Evaluating PI-CFO vector field "
        "on ground-truth states..."
    )

    q_t_pi = (
        model_vector_field_on_ground_truth(
            pi_method,
            pi_state,
            q_true,
            bathymetry,
            time,
            batch_size=batch_size,
        )
    )

    residual_cfo = residual_time_series(
        q_true,
        q_t_cfo,
        bathymetry,
        dx=dx,
        dy=dy,
        gravity=gravity,
    )

    residual_pi = residual_time_series(
        q_true,
        q_t_pi,
        bathymetry,
        dx=dx,
        dy=dy,
        gravity=gravity,
    )

    fig = plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.semilogy(
        time,
        residual_cfo,
        label="Bathy-CFO",
    )

    plt.semilogy(
        time,
        residual_pi,
        label="Bathy-PI-CFO",
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Variable-bottom SWE residual MSE"
    )

    plt.title(
        "Plot 7B — Same-ground-truth vector-field residual"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        output_dir
        /
        "07b_same_gt_vector_field_residual.png"
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        residual_cfo,
        residual_pi,
    )


# ================================================================
# PLOT 8
# MASS DRIFT
# ================================================================

def plot_08_mass_drift(
    *,
    time,
    q_true,
    q_cfo,
    q_pi,
    dx,
    dy,
    output_dir,
):

    drift_true = (
        mass_drift_time_series(
            q_true,
            dx=dx,
            dy=dy,
        )
    )

    drift_cfo = (
        mass_drift_time_series(
            q_cfo,
            dx=dx,
            dy=dy,
        )
    )

    drift_pi = (
        mass_drift_time_series(
            q_pi,
            dx=dx,
            dy=dy,
        )
    )

    fig = plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        time,
        drift_true,
        label="PyClaw",
    )

    plt.plot(
        time,
        drift_cfo,
        label="Bathy-CFO",
    )

    plt.plot(
        time,
        drift_pi,
        label="Bathy-PI-CFO",
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        "Mean relative mass drift"
    )

    plt.title(
        "Plot 8 — Mass conservation through time"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    path = (
        output_dir
        /
        "08_mass_drift_vs_time.png"
    )

    fig.savefig(
        path,
        dpi=220,
    )

    plt.close(
        fig
    )

    return (
        drift_true,
        drift_cfo,
        drift_pi,
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    dataset_path = resolve_path(
        args.dataset_path
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

    output_dir = (
        output_dir
        /
        args.split
        /
        f"trajectory_{args.trajectory}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print(
        "\nLoading dataset..."
    )

    dataset = (
        load_bathymetry_dataset(
            dataset_path
        )
    )

    split = require_split(
        dataset,
        args.split,
    )

    q_true = np.asarray(
        split[
            "q"
        ],
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    time = np.asarray(
        split[
            "time"
        ],
        dtype=np.float32,
    )

    x = np.asarray(
        dataset[
            "x"
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        dataset[
            "y"
        ],
        dtype=np.float32,
    )

    if not (
        0
        <=
        args.trajectory
        <
        q_true.shape[
            0
        ]
    ):

        raise ValueError(
            f"trajectory must be between "
            f"0 and {q_true.shape[0] - 1}."
        )

    input_shape = tuple(
        q_true.shape[
            2:
        ]
    )

    condition_shape = tuple(
        bathymetry.shape[
            1:
        ]
    )

    print(
        "\n========================================"
    )

    print(
        "BATHYMETRY PLOT DATA"
    )

    print(
        "========================================"
    )

    print(
        f"split: {args.split}"
    )

    print(
        f"q shape: {q_true.shape}"
    )

    print(
        f"bathymetry shape: "
        f"{bathymetry.shape}"
    )

    print(
        f"time: {time[0]:.3f} "
        f"-> {time[-1]:.3f}"
    )

    print(
        f"trajectory: "
        f"{args.trajectory}"
    )

    print(
        f"output: "
        f"{output_dir}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # RESTORE CFO
    # ============================================================

    print(
        "\nRestoring Bathy-CFO..."
    )

    cfo_method = (
        build_cfo_method(
            input_shape,
            condition_shape,
            args,
        )
    )

    cfo_state = restore_method(
        cfo_method,
        ckpt_dir=cfo_ckpt_dir,
        prefix=args.cfo_prefix,
        seed=args.seed,
    )

    print(
        "Bathy-CFO restored."
    )

    # ============================================================
    # RESTORE PI-CFO
    # ============================================================

    print(
        "\nRestoring Bathy-PI-CFO..."
    )

    pi_method = (
        build_pi_method(
            input_shape,
            condition_shape,
            args,
        )
    )

    pi_state = restore_method(
        pi_method,
        ckpt_dir=pi_ckpt_dir,
        prefix=args.pi_prefix,
        seed=args.seed,
    )

    print(
        "Bathy-PI-CFO restored."
    )

    # ============================================================
    # GENERATE ROLLOUTS
    # ============================================================

    print(
        "\nGenerating Bathy-CFO rollout..."
    )

    q_cfo = (
        cfo_method.uniform_inference(
            cfo_state,
            q_true[
                :,
                0,
            ],
            trajectory_points_num=(
                q_true.shape[
                    1
                ]
            ),
            steps_per_segment=(
                args.steps_per_segment
            ),
            condition=bathymetry,
            method="RK4",
        )
    )

    q_cfo = np.asarray(
        q_cfo,
        dtype=np.float32,
    )

    print(
        "Generating Bathy-PI-CFO rollout..."
    )

    q_pi = (
        pi_method.uniform_inference(
            pi_state,
            q_true[
                :,
                0,
            ],
            trajectory_points_num=(
                q_true.shape[
                    1
                ]
            ),
            steps_per_segment=(
                args.steps_per_segment
            ),
            condition=bathymetry,
            method="RK4",
        )
    )

    q_pi = np.asarray(
        q_pi,
        dtype=np.float32,
    )

    # ============================================================
    # SANITY CHECKS
    # ============================================================

    if not np.all(
        np.isfinite(
            q_cfo
        )
    ):

        raise RuntimeError(
            "CFO rollout contains NaN/Inf."
        )

    if not np.all(
        np.isfinite(
            q_pi
        )
    ):

        raise RuntimeError(
            "PI-CFO rollout contains NaN/Inf."
        )

    print(
        "\n========================================"
    )

    print(
        "ROLLOUT SANITY CHECK"
    )

    print(
        "========================================"
    )

    print(
        "CFO Relative L2:    "
        f"{relative_l2(q_true, q_cfo):.8f}"
    )

    print(
        "PI-CFO Relative L2: "
        f"{relative_l2(q_true, q_pi):.8f}"
    )

    print(
        "Truth minimum h:    "
        f"{np.min(q_true[..., 0]):.8f}"
    )

    print(
        "CFO minimum h:      "
        f"{np.min(q_cfo[..., 0]):.8f}"
    )

    print(
        "PI minimum h:       "
        f"{np.min(q_pi[..., 0]):.8f}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # PLOT 1
    # ============================================================

    print(
        "\nCreating Plot 1..."
    )

    plot_01_bathymetry_free_surface(
        x=x,
        y=y,
        time=time,
        q_true=q_true,
        q_cfo=q_cfo,
        q_pi=q_pi,
        bathymetry=bathymetry,
        trajectory=args.trajectory,
        target_time=(
            args.free_surface_time
        ),
        output_dir=output_dir,
    )

    # ============================================================
    # PLOTS 2-6 AT REQUESTED TIMES
    # ============================================================

    for target_time in (
        args.snapshot_times
    ):

        print(
            f"\nCreating spatial plots "
            f"at t≈{target_time:.2f}..."
        )

        plot_02_field_comparison(
            x=x,
            y=y,
            time=time,
            q_true=q_true,
            q_cfo=q_cfo,
            q_pi=q_pi,
            bathymetry=bathymetry,
            trajectory=(
                args.trajectory
            ),
            target_time=target_time,
            output_dir=output_dir,
        )

        plot_03_error_maps(
            x=x,
            y=y,
            time=time,
            q_true=q_true,
            q_cfo=q_cfo,
            q_pi=q_pi,
            bathymetry=bathymetry,
            trajectory=(
                args.trajectory
            ),
            target_time=target_time,
            output_dir=output_dir,
        )

        plot_04_velocity_vectors(
            x=x,
            y=y,
            time=time,
            q_true=q_true,
            q_cfo=q_cfo,
            q_pi=q_pi,
            bathymetry=bathymetry,
            trajectory=(
                args.trajectory
            ),
            target_time=target_time,
            skip=(
                args.quiver_skip
            ),
            quiver_scale=(
                args.quiver_scale
            ),
            output_dir=output_dir,
        )

        plot_05_velocity_magnitude(
            x=x,
            y=y,
            time=time,
            q_true=q_true,
            q_cfo=q_cfo,
            q_pi=q_pi,
            bathymetry=bathymetry,
            trajectory=(
                args.trajectory
            ),
            target_time=target_time,
            output_dir=output_dir,
        )

        plot_06_cross_sections(
            x=x,
            y=y,
            time=time,
            q_true=q_true,
            q_cfo=q_cfo,
            q_pi=q_pi,
            bathymetry=bathymetry,
            trajectory=(
                args.trajectory
            ),
            target_time=target_time,
            output_dir=output_dir,
        )

    # ============================================================
    # PLOT 7A
    # ============================================================

    print(
        "\nCreating Plot 7A: "
        "own-rollout SWE residual..."
    )

    (
        residual_true,
        residual_cfo_rollout,
        residual_pi_rollout,
    ) = plot_07a_rollout_residual(
        time=time,
        q_true=q_true,
        q_cfo=q_cfo,
        q_pi=q_pi,
        bathymetry=bathymetry,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
        output_dir=output_dir,
    )

    # ============================================================
    # PLOT 7B
    # ============================================================

    print(
        "\nCreating Plot 7B: "
        "same-ground-truth vector field..."
    )

    (
        residual_cfo_gt,
        residual_pi_gt,
    ) = plot_07b_same_gt_vector_field(
        time=time,
        q_true=q_true,
        bathymetry=bathymetry,
        cfo_method=cfo_method,
        cfo_state=cfo_state,
        pi_method=pi_method,
        pi_state=pi_state,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
        batch_size=(
            args.vector_field_batch_size
        ),
        output_dir=output_dir,
    )

    # ============================================================
    # PLOT 8
    # ============================================================

    print(
        "\nCreating Plot 8: "
        "mass conservation..."
    )

    (
        mass_true,
        mass_cfo,
        mass_pi,
    ) = plot_08_mass_drift(
        time=time,
        q_true=q_true,
        q_cfo=q_cfo,
        q_pi=q_pi,
        dx=args.dx,
        dy=args.dy,
        output_dir=output_dir,
    )

    # ============================================================
    # SAVE NUMERICAL CURVES
    # ============================================================

    np.savez(
        output_dir
        /
        "diagnostic_curves.npz",

        time=time,

        rollout_residual_true=(
            residual_true
        ),

        rollout_residual_cfo=(
            residual_cfo_rollout
        ),

        rollout_residual_pi=(
            residual_pi_rollout
        ),

        same_gt_residual_cfo=(
            residual_cfo_gt
        ),

        same_gt_residual_pi=(
            residual_pi_gt
        ),

        mass_drift_true=(
            mass_true
        ),

        mass_drift_cfo=(
            mass_cfo
        ),

        mass_drift_pi=(
            mass_pi
        ),
    )

    # ============================================================
    # PRINT PHYSICS SUMMARY
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "PHYSICS DIAGNOSTIC SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "\nMean own-rollout SWE residual:"
    )

    print(
        "PyClaw FD: "
        f"{np.mean(residual_true):.8e}"
    )

    print(
        "CFO:       "
        f"{np.mean(residual_cfo_rollout):.8e}"
    )

    print(
        "PI-CFO:    "
        f"{np.mean(residual_pi_rollout):.8e}"
    )

    print(
        "\nMean same-GT vector-field residual:"
    )

    print(
        "CFO:       "
        f"{np.mean(residual_cfo_gt):.8e}"
    )

    print(
        "PI-CFO:    "
        f"{np.mean(residual_pi_gt):.8e}"
    )

    improvement = (
        100.0
        *
        (
            np.mean(
                residual_cfo_gt
            )
            -
            np.mean(
                residual_pi_gt
            )
        )
        /
        max(
            np.mean(
                residual_cfo_gt
            ),
            1e-12,
        )
    )

    print(
        "\nPI-CFO same-GT residual "
        "improvement over CFO:"
    )

    print(
        f"{improvement:.2f}%"
    )

    print(
        "\nFinal mean relative mass drift:"
    )

    print(
        "PyClaw: "
        f"{mass_true[-1]:.8e}"
    )

    print(
        "CFO:    "
        f"{mass_cfo[-1]:.8e}"
    )

    print(
        "PI-CFO: "
        f"{mass_pi[-1]:.8e}"
    )

    print(
        "\nAll figures saved to:"
    )

    print(
        output_dir
    )

    print(
        "========================================"
    )

    # ============================================================
    # CLEANUP
    # ============================================================

    del cfo_state
    del pi_state
    del cfo_method
    del pi_method

    gc.collect()

    try:

        jax.clear_caches()

    except AttributeError:

        pass


if __name__ == "__main__":

    main()