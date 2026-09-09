"""
Comprehensive evaluation for Full-State CFO vs Physics-Informed CFO.

Outputs
-------
results/
    cfo_predictions.npy
    picfo_predictions.npy
    test_ground_truth.npy

    phase37_results.csv
    phase37_results.md
    evaluation_summary.txt

    error_vs_time.csv
    error_vs_time.png

    mass_drift_vs_time.csv
    mass_drift_vs_time.png

    mass_error_vs_ground_truth.csv
    mass_error_vs_ground_truth.png

    pde_residual_vs_time.csv
    pde_residual_vs_time.png

    field_comparison_traj00_h.png
    field_comparison_traj00_hu.png
    field_comparison_traj00_hv.png
    ...

This script assumes the state layout

    q = [h, hu, hv]

with array shape

    (batch, time, nx, ny, 3).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


# ================================================================
# QUIET UNNECESSARY TENSORFLOW / ABSL OUTPUT
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
# IMPORTS
# ================================================================

import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cfo import ContinuousFlowOperator
from pi_cfo import PhysicsInformedCFO

from models.factory import build_model

from train import init_cfo_train_state

from utils.checkpoints import load_train_state

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)

from utils.physics_swe import swe_residual

from utils.readers import load_dataset_splits


# ================================================================
# CONSTANTS
# ================================================================

CHANNEL_NAMES = (
    "h",
    "hu",
    "hv",
)

EPS = 1e-12


# ================================================================
# COMMAND-LINE ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Full-State CFO and PI-CFO "
            "on the shallow-water test set."
        )
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default="swe_full_small",
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=[
            "train",
            "eval",
            "test",
        ],
    )

    # ------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="UNet2D",
    )

    parser.add_argument(
        "--spline-type",
        type=str,
        default="quintic",
        choices=[
            "linear",
            "quintic",
        ],
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-5,
    )

    # ------------------------------------------------------------
    # CFO checkpoint
    #
    # IMPORTANT:
    #
    # ckpt-dir may point directly to a run, e.g.
    #
    # checkpoints/best/fno_cfo_e200_seed0
    #
    # Therefore the default prefix is intentionally empty.
    # ------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default="checkpoints/best",
    )

    parser.add_argument(
        "--cfo-ckpt-prefix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--cfo-ckpt-step",
        type=int,
        default=None,
    )

    # ------------------------------------------------------------
    # PI-CFO checkpoint
    # ------------------------------------------------------------

    parser.add_argument(
        "--picfo-ckpt-dir",
        type=str,
        default="checkpoints/best",
    )

    parser.add_argument(
        "--picfo-ckpt-prefix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--picfo-ckpt-step",
        type=int,
        default=None,
    )

    # ------------------------------------------------------------
    # PI-CFO physics settings
    # ------------------------------------------------------------

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=1e-3,
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

    # ------------------------------------------------------------
    # Optimizer settings
    #
    # These are used only to create the TrainState structure before
    # restoring the stored parameters / optimizer state.
    # ------------------------------------------------------------

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.99,
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------

    parser.add_argument(
        "--solver",
        type=str,
        default="RK4",
        choices=[
            "RK4",
            "Euler",
            "Heun",
        ],
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    # ------------------------------------------------------------
    # Field-comparison figures
    # ------------------------------------------------------------

    parser.add_argument(
        "--n-field-trajectories",
        type=int,
        default=2,
        help=(
            "Number of test trajectories for which "
            "spatial field-comparison figures are created."
        ),
    )

    parser.add_argument(
        "--field-times",
        type=float,
        nargs="+",
        default=[
            0.25,
            0.50,
            0.75,
            1.00,
        ],
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
    )

    return parser.parse_args()


# ================================================================
# CHECKPOINT PATH HELPERS
# ================================================================

def checkpoint_root(
    ckpt_dir: str,
    prefix: str,
) -> Path:
    """Return the effective checkpoint-manager directory."""

    root = Path(
        ckpt_dir
    ).resolve()

    if prefix:
        root = root / prefix

    return root


# ================================================================
# BASIC METRICS
# ================================================================

def relative_l2_np(
    target: np.ndarray,
    pred: np.ndarray,
) -> float:
    """Global relative L2 error."""

    numerator = np.linalg.norm(
        pred.reshape(-1)
        -
        target.reshape(-1)
    )

    denominator = np.linalg.norm(
        target.reshape(-1)
    )

    return float(
        numerator
        /
        (
            denominator
            +
            EPS
        )
    )


def channel_relative_l2(
    target: np.ndarray,
    pred: np.ndarray,
    channel: int,
) -> float:
    """Relative L2 error for one SWE state channel."""

    return relative_l2_np(
        target[
            ...,
            channel
        ],
        pred[
            ...,
            channel
        ],
    )


def relative_error_per_trajectory_and_time(
    target: np.ndarray,
    pred: np.ndarray,
) -> np.ndarray:
    """Relative L2 at every trajectory and time.

    Returns
    -------
    errors:
        Shape (batch, time)
    """

    difference = (
        pred
        -
        target
    )

    numerator = np.sqrt(
        np.sum(
            difference**2,
            axis=(
                2,
                3,
                4,
            ),
        )
    )

    denominator = np.sqrt(
        np.sum(
            target**2,
            axis=(
                2,
                3,
                4,
            ),
        )
    )

    return (
        numerator
        /
        (
            denominator
            +
            EPS
        )
    )


# ================================================================
# CHECKPOINT RESTORATION
# ================================================================

def restore_cfo(
    *,
    input_shape,
    args,
):
    """Restore ordinary CFO checkpoint."""

    model = build_model(
        args.model,
        input_shape,
        use_condition=False,
    )

    method = ContinuousFlowOperator(
        model=model,
        input_shape=input_shape,
        gamma=args.gamma,
        spline_type=args.spline_type,
        use_condition=False,
    )

    state_template = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    ckpt_dir = str(
        Path(
            args.cfo_ckpt_dir
        ).resolve()
    )

    effective_root = checkpoint_root(
        ckpt_dir,
        args.cfo_ckpt_prefix,
    )

    print(
        "CFO checkpoint root:"
    )

    print(
        effective_root
    )

    if not effective_root.exists():
        raise FileNotFoundError(
            "CFO checkpoint directory does not exist:\n"
            f"{effective_root}"
        )

    state = load_train_state(
        target_state=state_template,
        ckpt_dir=ckpt_dir,
        prefix=args.cfo_ckpt_prefix,
        step=args.cfo_ckpt_step,
    )

    return (
        method,
        state,
    )


def restore_picfo(
    *,
    input_shape,
    args,
):
    """Restore Physics-Informed CFO checkpoint."""

    model = build_model(
        args.model,
        input_shape,
        use_condition=False,
    )

    method = PhysicsInformedCFO(
        model=model,
        input_shape=input_shape,
        gamma=args.gamma,
        spline_type=args.spline_type,
        use_condition=False,
        lambda_pde=args.lambda_pde,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    state_template = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    ckpt_dir = str(
        Path(
            args.picfo_ckpt_dir
        ).resolve()
    )

    effective_root = checkpoint_root(
        ckpt_dir,
        args.picfo_ckpt_prefix,
    )

    print(
        "PI-CFO checkpoint root:"
    )

    print(
        effective_root
    )

    if not effective_root.exists():
        raise FileNotFoundError(
            "PI-CFO checkpoint directory does not exist:\n"
            f"{effective_root}"
        )

    state = load_train_state(
        target_state=state_template,
        ckpt_dir=ckpt_dir,
        prefix=args.picfo_ckpt_prefix,
        step=args.picfo_ckpt_step,
    )

    return (
        method,
        state,
    )


# ================================================================
# MODEL ROLLOUT
# ================================================================

def rollout(
    method,
    state,
    target,
    args,
) -> np.ndarray:
    """Perform complete continuous-time rollout."""

    prediction = method.uniform_inference(
        state,
        target[
            :,
            0
        ],
        trajectory_points_num=(
            target.shape[1]
        ),
        steps_per_segment=(
            args.steps_per_segment
        ),
        method=args.solver,
    )

    return np.asarray(
        prediction,
        dtype=np.float32,
    )


# ================================================================
# PHASE 35 — ERROR VS TIME
# ================================================================

def compute_error_vs_time(
    target,
    cfo_pred,
    picfo_pred,
):
    """Compute average relative error at each time."""

    cfo_all = (
        relative_error_per_trajectory_and_time(
            target,
            cfo_pred,
        )
    )

    picfo_all = (
        relative_error_per_trajectory_and_time(
            target,
            picfo_pred,
        )
    )

    return {
        "cfo_mean":
            np.mean(
                cfo_all,
                axis=0,
            ),

        "cfo_std":
            np.std(
                cfo_all,
                axis=0,
            ),

        "picfo_mean":
            np.mean(
                picfo_all,
                axis=0,
            ),

        "picfo_std":
            np.std(
                picfo_all,
                axis=0,
            ),
    }


def save_error_vs_time(
    *,
    times,
    error_data,
    results_dir,
):
    """Save error-vs-time CSV and figure."""

    csv_path = (
        results_dir
        /
        "error_vs_time.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "time",
                "cfo_mean_rel_l2",
                "cfo_std_rel_l2",
                "picfo_mean_rel_l2",
                "picfo_std_rel_l2",
            ]
        )

        for i, t in enumerate(
            times
        ):

            writer.writerow(
                [
                    float(t),

                    float(
                        error_data[
                            "cfo_mean"
                        ][i]
                    ),

                    float(
                        error_data[
                            "cfo_std"
                        ][i]
                    ),

                    float(
                        error_data[
                            "picfo_mean"
                        ][i]
                    ),

                    float(
                        error_data[
                            "picfo_std"
                        ][i]
                    ),
                ]
            )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.0,
        )
    )

    ax.plot(
        times,
        error_data[
            "cfo_mean"
        ],
        label="CFO",
        linewidth=2,
    )

    ax.plot(
        times,
        error_data[
            "picfo_mean"
        ],
        label="PI-CFO",
        linewidth=2,
    )

    ax.fill_between(
        times,
        error_data[
            "cfo_mean"
        ]
        -
        error_data[
            "cfo_std"
        ],
        error_data[
            "cfo_mean"
        ]
        +
        error_data[
            "cfo_std"
        ],
        alpha=0.15,
    )

    ax.fill_between(
        times,
        error_data[
            "picfo_mean"
        ]
        -
        error_data[
            "picfo_std"
        ],
        error_data[
            "picfo_mean"
        ]
        +
        error_data[
            "picfo_std"
        ],
        alpha=0.15,
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Relative L2 error"
    )

    ax.set_title(
        "Prediction Error vs Time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        /
        "error_vs_time.png",
        dpi=300,
    )

    plt.close(
        fig
    )


# ================================================================
# PHASE 34 — MASS DRIFT
# ================================================================

def compute_total_mass(
    q: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Compute domain-integrated water mass.

    q shape:
        (batch, time, nx, ny, channels)

    Returns
    -------
    mass:
        shape (batch, time)
    """

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
        dx
        *
        dy
    )

    return mass


def relative_mass_drift(
    mass: np.ndarray,
) -> np.ndarray:
    """Relative drift from initial total mass."""

    initial = mass[
        :,
        0:1,
    ]

    return (
        np.abs(
            mass
            -
            initial
        )
        /
        (
            np.abs(
                initial
            )
            +
            EPS
        )
    )


def relative_mass_error_vs_truth(
    true_mass,
    predicted_mass,
):
    """Mass prediction error relative to true trajectory mass."""

    return (
        np.abs(
            predicted_mass
            -
            true_mass
        )
        /
        (
            np.abs(
                true_mass
            )
            +
            EPS
        )
    )


def save_mass_outputs(
    *,
    target,
    cfo_pred,
    picfo_pred,
    times,
    dx,
    dy,
    results_dir,
):
    """Compute and save mass diagnostics."""

    true_mass = compute_total_mass(
        target,
        dx,
        dy,
    )

    cfo_mass = compute_total_mass(
        cfo_pred,
        dx,
        dy,
    )

    picfo_mass = compute_total_mass(
        picfo_pred,
        dx,
        dy,
    )

    true_drift = relative_mass_drift(
        true_mass
    )

    cfo_drift = relative_mass_drift(
        cfo_mass
    )

    picfo_drift = relative_mass_drift(
        picfo_mass
    )

    true_drift_mean = np.mean(
        true_drift,
        axis=0,
    )

    cfo_drift_mean = np.mean(
        cfo_drift,
        axis=0,
    )

    picfo_drift_mean = np.mean(
        picfo_drift,
        axis=0,
    )

    # ------------------------------------------------------------
    # CSV: mass drift
    # ------------------------------------------------------------

    csv_path = (
        results_dir
        /
        "mass_drift_vs_time.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time",
                "ground_truth_mean_mass_drift",
                "cfo_mean_mass_drift",
                "picfo_mean_mass_drift",
            ]
        )

        for i, t in enumerate(
            times
        ):

            writer.writerow(
                [
                    float(t),

                    float(
                        true_drift_mean[
                            i
                        ]
                    ),

                    float(
                        cfo_drift_mean[
                            i
                        ]
                    ),

                    float(
                        picfo_drift_mean[
                            i
                        ]
                    ),
                ]
            )

    # ------------------------------------------------------------
    # Mass drift plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.0,
        )
    )

    ax.plot(
        times,
        true_drift_mean,
        label="Ground Truth",
        linewidth=2,
    )

    ax.plot(
        times,
        cfo_drift_mean,
        label="CFO",
        linewidth=2,
    )

    ax.plot(
        times,
        picfo_drift_mean,
        label="PI-CFO",
        linewidth=2,
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Relative mass drift"
    )

    ax.set_title(
        "Mass Drift vs Time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        /
        "mass_drift_vs_time.png",
        dpi=300,
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Mass error relative to ground truth
    # ------------------------------------------------------------

    cfo_mass_error = (
        relative_mass_error_vs_truth(
            true_mass,
            cfo_mass,
        )
    )

    picfo_mass_error = (
        relative_mass_error_vs_truth(
            true_mass,
            picfo_mass,
        )
    )

    cfo_mass_error_mean = np.mean(
        cfo_mass_error,
        axis=0,
    )

    picfo_mass_error_mean = np.mean(
        picfo_mass_error,
        axis=0,
    )

    csv_path = (
        results_dir
        /
        "mass_error_vs_ground_truth.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time",
                "cfo_mass_error",
                "picfo_mass_error",
            ]
        )

        for i, t in enumerate(
            times
        ):

            writer.writerow(
                [
                    float(t),

                    float(
                        cfo_mass_error_mean[
                            i
                        ]
                    ),

                    float(
                        picfo_mass_error_mean[
                            i
                        ]
                    ),
                ]
            )

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.0,
        )
    )

    ax.plot(
        times,
        cfo_mass_error_mean,
        label="CFO",
        linewidth=2,
    )

    ax.plot(
        times,
        picfo_mass_error_mean,
        label="PI-CFO",
        linewidth=2,
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Relative mass error"
    )

    ax.set_title(
        "Mass Error Relative to Ground Truth"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        /
        "mass_error_vs_ground_truth.png",
        dpi=300,
    )

    plt.close(
        fig
    )

    return {
        "true_mass":
            true_mass,

        "cfo_mass":
            cfo_mass,

        "picfo_mass":
            picfo_mass,

        "true_drift":
            true_drift,

        "cfo_drift":
            cfo_drift,

        "picfo_drift":
            picfo_drift,

        "cfo_mass_error":
            cfo_mass_error,

        "picfo_mass_error":
            picfo_mass_error,
    }


# ================================================================
# PHASE 33 — SWE PDE RESIDUAL
# ================================================================

def temporal_derivative(
    trajectory: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Finite-difference temporal derivative."""

    edge_order = (
        2
        if trajectory.shape[1] >= 3
        else 1
    )

    return np.gradient(
        trajectory,
        times,
        axis=1,
        edge_order=edge_order,
    )


def compute_pde_residual_vs_time(
    trajectory: np.ndarray,
    times: np.ndarray,
    dx: float,
    dy: float,
    gravity: float,
):
    """Compute SWE residual MSE at every time."""

    q_t = temporal_derivative(
        trajectory,
        times,
    )

    n_time = (
        trajectory.shape[1]
    )

    overall = np.zeros(
        n_time,
        dtype=np.float64,
    )

    channel = np.zeros(
        (
            n_time,
            3,
        ),
        dtype=np.float64,
    )

    for t_idx in range(
        n_time
    ):

        q_now = jnp.asarray(
            trajectory[
                :,
                t_idx,
            ]
        )

        qt_now = jnp.asarray(
            q_t[
                :,
                t_idx,
            ]
        )

        residual = swe_residual(
            q=q_now,
            q_t=qt_now,
            dx=dx,
            dy=dy,
            g=gravity,
        )

        residual = np.asarray(
            residual
        )

        overall[
            t_idx
        ] = np.mean(
            residual**2
        )

        for channel_idx in range(
            3
        ):

            channel[
                t_idx,
                channel_idx,
            ] = np.mean(
                residual[
                    ...,
                    channel_idx
                ]**2
            )

    return (
        overall,
        channel,
    )


def save_pde_outputs(
    *,
    target,
    cfo_pred,
    picfo_pred,
    times,
    dx,
    dy,
    gravity,
    results_dir,
):
    """Compute and save PDE residual diagnostics."""

    print(
        "Computing ground-truth PDE residual..."
    )

    (
        truth_overall,
        truth_channel,
    ) = compute_pde_residual_vs_time(
        target,
        times,
        dx,
        dy,
        gravity,
    )

    print(
        "Computing CFO PDE residual..."
    )

    (
        cfo_overall,
        cfo_channel,
    ) = compute_pde_residual_vs_time(
        cfo_pred,
        times,
        dx,
        dy,
        gravity,
    )

    print(
        "Computing PI-CFO PDE residual..."
    )

    (
        picfo_overall,
        picfo_channel,
    ) = compute_pde_residual_vs_time(
        picfo_pred,
        times,
        dx,
        dy,
        gravity,
    )

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------

    csv_path = (
        results_dir
        /
        "pde_residual_vs_time.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time",

                "ground_truth_pde_mse",
                "cfo_pde_mse",
                "picfo_pde_mse",

                "cfo_mass_residual_mse",
                "cfo_xmomentum_residual_mse",
                "cfo_ymomentum_residual_mse",

                "picfo_mass_residual_mse",
                "picfo_xmomentum_residual_mse",
                "picfo_ymomentum_residual_mse",
            ]
        )

        for i, t in enumerate(
            times
        ):

            writer.writerow(
                [
                    float(t),

                    float(
                        truth_overall[
                            i
                        ]
                    ),

                    float(
                        cfo_overall[
                            i
                        ]
                    ),

                    float(
                        picfo_overall[
                            i
                        ]
                    ),

                    float(
                        cfo_channel[
                            i,
                            0
                        ]
                    ),

                    float(
                        cfo_channel[
                            i,
                            1
                        ]
                    ),

                    float(
                        cfo_channel[
                            i,
                            2
                        ]
                    ),

                    float(
                        picfo_channel[
                            i,
                            0
                        ]
                    ),

                    float(
                        picfo_channel[
                            i,
                            1
                        ]
                    ),

                    float(
                        picfo_channel[
                            i,
                            2
                        ]
                    ),
                ]
            )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            7.5,
            5.0,
        )
    )

    ax.plot(
        times,
        truth_overall,
        label="Ground Truth",
        linewidth=2,
    )

    ax.plot(
        times,
        cfo_overall,
        label="CFO",
        linewidth=2,
    )

    ax.plot(
        times,
        picfo_overall,
        label="PI-CFO",
        linewidth=2,
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Mean squared SWE residual"
    )

    ax.set_title(
        "SWE Residual vs Time"
    )

    ax.set_yscale(
        "log"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        results_dir
        /
        "pde_residual_vs_time.png",
        dpi=300,
    )

    plt.close(
        fig
    )

    return {
        "truth_overall":
            truth_overall,

        "cfo_overall":
            cfo_overall,

        "picfo_overall":
            picfo_overall,

        "truth_channel":
            truth_channel,

        "cfo_channel":
            cfo_channel,

        "picfo_channel":
            picfo_channel,
    }


# ================================================================
# PHASE 36 — SPATIAL FIELD COMPARISONS
# ================================================================

def nearest_time_indices(
    times,
    desired_times,
):
    """Find stored time index nearest each requested time."""

    indices = []

    for desired in desired_times:

        idx = int(
            np.argmin(
                np.abs(
                    times
                    -
                    desired
                )
            )
        )

        if idx not in indices:
            indices.append(
                idx
            )

    return indices


def save_field_comparison(
    *,
    target,
    cfo_pred,
    picfo_pred,
    times,
    trajectory_index,
    channel_index,
    requested_times,
    results_dir,
):
    """Create spatial field-comparison figure."""

    time_indices = nearest_time_indices(
        times,
        requested_times,
    )

    n_rows = len(
        time_indices
    )

    fig, axes = plt.subplots(
        n_rows,
        5,
        figsize=(
            17,
            3.5 * n_rows,
        ),
        squeeze=False,
    )

    channel_name = (
        CHANNEL_NAMES[
            channel_index
        ]
    )

    column_titles = [
        "Ground Truth",
        "CFO",
        "PI-CFO",
        "|CFO - Truth|",
        "|PI-CFO - Truth|",
    ]

    for col_idx, title in enumerate(
        column_titles
    ):

        axes[
            0,
            col_idx
        ].set_title(
            title,
            fontsize=12,
        )

    for row_idx, time_idx in enumerate(
        time_indices
    ):

        truth_field = (
            target[
                trajectory_index,
                time_idx,
                ...,
                channel_index,
            ]
        )

        cfo_field = (
            cfo_pred[
                trajectory_index,
                time_idx,
                ...,
                channel_index,
            ]
        )

        picfo_field = (
            picfo_pred[
                trajectory_index,
                time_idx,
                ...,
                channel_index,
            ]
        )

        cfo_error = np.abs(
            cfo_field
            -
            truth_field
        )

        picfo_error = np.abs(
            picfo_field
            -
            truth_field
        )

        state_min = min(
            float(
                np.min(
                    truth_field
                )
            ),
            float(
                np.min(
                    cfo_field
                )
            ),
            float(
                np.min(
                    picfo_field
                )
            ),
        )

        state_max = max(
            float(
                np.max(
                    truth_field
                )
            ),
            float(
                np.max(
                    cfo_field
                )
            ),
            float(
                np.max(
                    picfo_field
                )
            ),
        )

        error_max = max(
            float(
                np.max(
                    cfo_error
                )
            ),
            float(
                np.max(
                    picfo_error
                )
            ),
            EPS,
        )

        im_truth = axes[
            row_idx,
            0
        ].imshow(
            truth_field,
            origin="lower",
            vmin=state_min,
            vmax=state_max,
        )

        axes[
            row_idx,
            1
        ].imshow(
            cfo_field,
            origin="lower",
            vmin=state_min,
            vmax=state_max,
        )

        axes[
            row_idx,
            2
        ].imshow(
            picfo_field,
            origin="lower",
            vmin=state_min,
            vmax=state_max,
        )

        im_error = axes[
            row_idx,
            3
        ].imshow(
            cfo_error,
            origin="lower",
            vmin=0.0,
            vmax=error_max,
        )

        axes[
            row_idx,
            4
        ].imshow(
            picfo_error,
            origin="lower",
            vmin=0.0,
            vmax=error_max,
        )

        axes[
            row_idx,
            0
        ].set_ylabel(
            (
                f"t={times[time_idx]:.2f}\n"
                f"{channel_name}"
            ),
            fontsize=11,
        )

        for col_idx in range(
            5
        ):

            axes[
                row_idx,
                col_idx
            ].set_xticks(
                []
            )

            axes[
                row_idx,
                col_idx
            ].set_yticks(
                []
            )

        fig.colorbar(
            im_truth,
            ax=axes[
                row_idx,
                0:3
            ].tolist(),
            fraction=0.018,
            pad=0.01,
        )

        fig.colorbar(
            im_error,
            ax=axes[
                row_idx,
                3:5
            ].tolist(),
            fraction=0.027,
            pad=0.01,
        )

    fig.suptitle(
        (
            f"Trajectory {trajectory_index}: "
            f"{channel_name} Field Comparison"
        ),
        fontsize=15,
    )

    fig.subplots_adjust(
        top=0.93,
        wspace=0.20,
        hspace=0.20,
    )

    output_path = (
        results_dir
        /
        (
            f"field_comparison_"
            f"traj{trajectory_index:02d}_"
            f"{channel_name}.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ================================================================
# PHASE 37 — FINAL RESULTS TABLE
# ================================================================

def build_phase37_results(
    *,
    target,
    cfo_pred,
    picfo_pred,
    mass_outputs,
    pde_outputs,
):
    """Build main comparison table."""

    cfo_rel_l2 = float(
        relative_L2_error(
            target,
            cfo_pred,
        )
    )

    cfo_rmse = float(
        rmse(
            target,
            cfo_pred,
        )
    )

    cfo_rel_fro = float(
        relative_frobenius_error(
            target,
            cfo_pred,
        )
    )

    picfo_rel_l2 = float(
        relative_L2_error(
            target,
            picfo_pred,
        )
    )

    picfo_rmse = float(
        rmse(
            target,
            picfo_pred,
        )
    )

    picfo_rel_fro = float(
        relative_frobenius_error(
            target,
            picfo_pred,
        )
    )

    cfo_channels = [
        channel_relative_l2(
            target,
            cfo_pred,
            i,
        )
        for i in range(
            3
        )
    ]

    picfo_channels = [
        channel_relative_l2(
            target,
            picfo_pred,
            i,
        )
        for i in range(
            3
        )
    ]

    cfo_pde = float(
        np.mean(
            pde_outputs[
                "cfo_overall"
            ]
        )
    )

    picfo_pde = float(
        np.mean(
            pde_outputs[
                "picfo_overall"
            ]
        )
    )

    cfo_mass_drift = float(
        np.mean(
            mass_outputs[
                "cfo_drift"
            ]
        )
    )

    picfo_mass_drift = float(
        np.mean(
            mass_outputs[
                "picfo_drift"
            ]
        )
    )

    cfo_final_mass_drift = float(
        np.mean(
            mass_outputs[
                "cfo_drift"
            ][
                :,
                -1
            ]
        )
    )

    picfo_final_mass_drift = float(
        np.mean(
            mass_outputs[
                "picfo_drift"
            ][
                :,
                -1
            ]
        )
    )

    results = [
        {
            "Model":
                "CFO",

            "Rel_L2_q":
                cfo_rel_l2,

            "E_h":
                cfo_channels[
                    0
                ],

            "E_hu":
                cfo_channels[
                    1
                ],

            "E_hv":
                cfo_channels[
                    2
                ],

            "PDE_residual_MSE":
                cfo_pde,

            "Mean_mass_drift":
                cfo_mass_drift,

            "Final_mass_drift":
                cfo_final_mass_drift,

            "RMSE":
                cfo_rmse,

            "Relative_Frobenius":
                cfo_rel_fro,
        },

        {
            "Model":
                "PI-CFO",

            "Rel_L2_q":
                picfo_rel_l2,

            "E_h":
                picfo_channels[
                    0
                ],

            "E_hu":
                picfo_channels[
                    1
                ],

            "E_hv":
                picfo_channels[
                    2
                ],

            "PDE_residual_MSE":
                picfo_pde,

            "Mean_mass_drift":
                picfo_mass_drift,

            "Final_mass_drift":
                picfo_final_mass_drift,

            "RMSE":
                picfo_rmse,

            "Relative_Frobenius":
                picfo_rel_fro,
        },
    ]

    return results


def save_phase37_results(
    results,
    results_dir,
):
    """Save Phase-37 table as CSV and Markdown."""

    columns = [
        "Model",
        "Rel_L2_q",
        "E_h",
        "E_hu",
        "E_hv",
        "PDE_residual_MSE",
        "Mean_mass_drift",
        "Final_mass_drift",
        "RMSE",
        "Relative_Frobenius",
    ]

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    csv_path = (
        results_dir
        /
        "phase37_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                row
            )

    # ------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------

    markdown_path = (
        results_dir
        /
        "phase37_results.md"
    )

    with markdown_path.open(
        "w"
    ) as f:

        f.write(
            "# Phase 37 — CFO vs PI-CFO Results\n\n"
        )

        f.write(
            "| Model | Rel. L2(q) | E_h | E_hu | E_hv | "
            "PDE Residual MSE | Mean Mass Drift |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|---:|---:|\n"
        )

        for row in results:

            f.write(
                "| "
                f"{row['Model']} | "
                f"{row['Rel_L2_q']:.6e} | "
                f"{row['E_h']:.6e} | "
                f"{row['E_hu']:.6e} | "
                f"{row['E_hv']:.6e} | "
                f"{row['PDE_residual_MSE']:.6e} | "
                f"{row['Mean_mass_drift']:.6e} |\n"
            )

    # ------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "PHASE 37 — CFO vs PI-CFO"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Model':<12}"
        f"{'Rel L2(q)':>14}"
        f"{'E_h':>14}"
        f"{'E_hu':>14}"
        f"{'E_hv':>14}"
        f"{'PDE MSE':>14}"
        f"{'Mass drift':>14}"
    )

    print(
        "-" * 96
    )

    for row in results:

        print(
            f"{row['Model']:<12}"
            f"{row['Rel_L2_q']:>14.5e}"
            f"{row['E_h']:>14.5e}"
            f"{row['E_hu']:>14.5e}"
            f"{row['E_hv']:>14.5e}"
            f"{row['PDE_residual_MSE']:>14.5e}"
            f"{row['Mean_mass_drift']:>14.5e}"
        )

    print(
        "============================================================"
    )


# ================================================================
# TEXT SUMMARY
# ================================================================

def save_summary(
    *,
    results,
    pde_outputs,
    mass_outputs,
    args,
    target_shape,
    results_dir,
):
    """Save human-readable evaluation summary."""

    path = (
        results_dir
        /
        "evaluation_summary.txt"
    )

    cfo = results[
        0
    ]

    picfo = results[
        1
    ]

    with path.open(
        "w"
    ) as f:

        f.write(
            "CFO vs PI-CFO Shallow-Water Evaluation\n"
        )

        f.write(
            "======================================\n\n"
        )

        f.write(
            f"Dataset: {args.dataset}\n"
        )

        f.write(
            f"Split: {args.split}\n"
        )

        f.write(
            f"Target shape: {target_shape}\n"
        )

        f.write(
            f"Model: {args.model}\n"
        )

        f.write(
            f"Solver: {args.solver}\n"
        )

        f.write(
            f"Steps per segment: "
            f"{args.steps_per_segment}\n"
        )

        f.write(
            f"lambda_PDE: "
            f"{args.lambda_pde}\n"
        )

        f.write(
            f"dx: {args.dx}\n"
        )

        f.write(
            f"dy: {args.dy}\n"
        )

        f.write(
            f"gravity: {args.gravity}\n\n"
        )

        f.write(
            "CFO checkpoint\n"
        )

        f.write(
            "--------------\n"
        )

        f.write(
            f"Directory: {args.cfo_ckpt_dir}\n"
        )

        f.write(
            f"Prefix: {args.cfo_ckpt_prefix}\n"
        )

        f.write(
            f"Step: {args.cfo_ckpt_step}\n\n"
        )

        f.write(
            "PI-CFO checkpoint\n"
        )

        f.write(
            "-----------------\n"
        )

        f.write(
            f"Directory: {args.picfo_ckpt_dir}\n"
        )

        f.write(
            f"Prefix: {args.picfo_ckpt_prefix}\n"
        )

        f.write(
            f"Step: {args.picfo_ckpt_step}\n\n"
        )

        f.write(
            "CFO\n"
        )

        f.write(
            "---\n"
        )

        for key, value in cfo.items():

            if key != "Model":

                f.write(
                    f"{key}: "
                    f"{value:.8e}\n"
                )

        f.write(
            "\nPI-CFO\n"
        )

        f.write(
            "------\n"
        )

        for key, value in picfo.items():

            if key != "Model":

                f.write(
                    f"{key}: "
                    f"{value:.8e}\n"
                )

        f.write(
            "\nRelative PI-CFO improvement over CFO\n"
        )

        f.write(
            "------------------------------------\n"
        )

        for metric in [
            "Rel_L2_q",
            "PDE_residual_MSE",
            "Mean_mass_drift",
        ]:

            cfo_value = cfo[
                metric
            ]

            picfo_value = picfo[
                metric
            ]

            improvement = (
                100.0
                *
                (
                    cfo_value
                    -
                    picfo_value
                )
                /
                (
                    abs(
                        cfo_value
                    )
                    +
                    EPS
                )
            )

            f.write(
                f"{metric}: "
                f"{improvement:.3f}%\n"
            )


# ================================================================
# MAIN
# ================================================================

def main():
    """Run complete CFO vs PI-CFO evaluation."""

    args = parse_args()

    # ============================================================
    # RESULTS DIRECTORY
    # ============================================================

    results_dir = Path(
        args.results_dir
    ).resolve()

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "CFO vs PI-CFO PHYSICS-AWARE EVALUATION"
    )

    print(
        "============================================================"
    )

    print(
        f"Results directory:\n"
        f"{results_dir}"
    )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print(
        "\nLoading dataset..."
    )

    splits = load_dataset_splits(
        args.dataset,
        file_path=args.dataset_path,
    )

    target = np.asarray(
        splits[
            args.split
        ],
        dtype=np.float32,
    )

    if target.ndim != 5:

        raise ValueError(
            "Expected SWE data shape "
            "(batch, time, nx, ny, 3), "
            f"but received {target.shape}."
        )

    if target.shape[
        -1
    ] != 3:

        raise ValueError(
            "Expected state channels [h, hu, hv], "
            f"but last dimension is {target.shape[-1]}."
        )

    input_shape = tuple(
        target.shape[
            2:
        ]
    )

    print(
        f"Target shape: "
        f"{target.shape}"
    )

    print(
        f"Input shape: "
        f"{input_shape}"
    )

    print(
        f"Number of trajectories: "
        f"{target.shape[0]}"
    )

    print(
        f"Time snapshots: "
        f"{target.shape[1]}"
    )

    # ============================================================
    # TIME GRID
    # ============================================================

    times = np.linspace(
        0.0,
        1.0,
        target.shape[
            1
        ],
        dtype=np.float64,
    )

    # ============================================================
    # RESTORE CFO
    # ============================================================

    print(
        "\nRestoring CFO checkpoint..."
    )

    (
        cfo_method,
        cfo_state,
    ) = restore_cfo(
        input_shape=input_shape,
        args=args,
    )

    print(
        "CFO checkpoint restored."
    )

    # ============================================================
    # RESTORE PI-CFO
    # ============================================================

    print(
        "\nRestoring PI-CFO checkpoint..."
    )

    (
        picfo_method,
        picfo_state,
    ) = restore_picfo(
        input_shape=input_shape,
        args=args,
    )

    print(
        "PI-CFO checkpoint restored."
    )

    # ============================================================
    # INFERENCE
    # ============================================================

    print(
        "\nRunning CFO inference..."
    )

    cfo_pred = rollout(
        cfo_method,
        cfo_state,
        target,
        args,
    )

    print(
        "CFO rollout complete."
    )

    print(
        "\nRunning PI-CFO inference..."
    )

    picfo_pred = rollout(
        picfo_method,
        picfo_state,
        target,
        args,
    )

    print(
        "PI-CFO rollout complete."
    )

    print(
        f"\nCFO prediction shape: "
        f"{cfo_pred.shape}"
    )

    print(
        f"PI-CFO prediction shape: "
        f"{picfo_pred.shape}"
    )

    # ============================================================
    # SAVE RAW ARRAYS
    # ============================================================

    np.save(
        results_dir
        /
        "test_ground_truth.npy",
        target,
    )

    np.save(
        results_dir
        /
        "cfo_predictions.npy",
        cfo_pred,
    )

    np.save(
        results_dir
        /
        "picfo_predictions.npy",
        picfo_pred,
    )

    # ============================================================
    # PHASE 35 — ERROR VS TIME
    # ============================================================

    print(
        "\nCreating error-vs-time outputs..."
    )

    error_data = compute_error_vs_time(
        target,
        cfo_pred,
        picfo_pred,
    )

    save_error_vs_time(
        times=times,
        error_data=error_data,
        results_dir=results_dir,
    )

    # ============================================================
    # PHASE 34 — MASS DRIFT
    # ============================================================

    print(
        "Creating mass-balance outputs..."
    )

    mass_outputs = save_mass_outputs(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        times=times,
        dx=args.dx,
        dy=args.dy,
        results_dir=results_dir,
    )

    # ============================================================
    # PHASE 33 — PDE RESIDUAL
    # ============================================================

    print(
        "Creating SWE residual outputs..."
    )

    pde_outputs = save_pde_outputs(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
        results_dir=results_dir,
    )

    # ============================================================
    # PHASE 36 — FIELD COMPARISONS
    # ============================================================

    print(
        "\nCreating Phase-36 field figures..."
    )

    n_field_trajectories = min(
        args.n_field_trajectories,
        target.shape[
            0
        ],
    )

    for trajectory_index in range(
        n_field_trajectories
    ):

        for channel_index in range(
            3
        ):

            save_field_comparison(
                target=target,
                cfo_pred=cfo_pred,
                picfo_pred=picfo_pred,
                times=times,
                trajectory_index=(
                    trajectory_index
                ),
                channel_index=(
                    channel_index
                ),
                requested_times=(
                    args.field_times
                ),
                results_dir=(
                    results_dir
                ),
            )

    # ============================================================
    # PHASE 37 — FINAL TABLE
    # ============================================================

    print(
        "Creating Phase-37 result table..."
    )

    results = build_phase37_results(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        mass_outputs=mass_outputs,
        pde_outputs=pde_outputs,
    )

    save_phase37_results(
        results,
        results_dir,
    )

    # ============================================================
    # TEXT SUMMARY
    # ============================================================

    save_summary(
        results=results,
        pde_outputs=pde_outputs,
        mass_outputs=mass_outputs,
        args=args,
        target_shape=target.shape,
        results_dir=results_dir,
    )

    # ============================================================
    # DONE
    # ============================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        f"All results saved to:\n"
        f"{results_dir}"
    )

    print(
        "\nMain files:"
    )

    print(
        "  phase37_results.csv"
    )

    print(
        "  phase37_results.md"
    )

    print(
        "  error_vs_time.png"
    )

    print(
        "  mass_drift_vs_time.png"
    )

    print(
        "  mass_error_vs_ground_truth.png"
    )

    print(
        "  pde_residual_vs_time.png"
    )

    print(
        "  field_comparison_trajXX_h.png"
    )

    print(
        "  field_comparison_trajXX_hu.png"
    )

    print(
        "  field_comparison_trajXX_hv.png"
    )

    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()