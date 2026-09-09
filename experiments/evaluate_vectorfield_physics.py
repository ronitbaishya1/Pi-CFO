from __future__ import annotations

"""
Training-consistent vector-field physics evaluation for CFO vs PI-CFO.

Why this script exists
----------------------

The normal evaluate_picfo.py script evaluates the SWE residual from
the completed RK4 rollout using a finite-difference time derivative:

    q_t ~= d q_rollout / dt

That is a useful rollout diagnostic, but it is not exactly the
quantity used inside PI-CFO training.

During PI-CFO training the neural operator itself predicts

    q_t = N_theta(t, q)

and the physics residual is built from

    R = N_theta(t, q) + div(F(q)).

This script evaluates that learned neural vector field directly.

It performs two comparisons:

1. Each model on its own rollout:
       CFO:
           N_CFO(t, q_CFO)

       PI-CFO:
           N_PI(t, q_PI)

2. Both models on exactly the same ground-truth states:
       N_CFO(t, q_GT)
       N_PI(t, q_GT)

The second comparison removes differences caused simply by CFO and
PI-CFO visiting slightly different rollout states.

Expected input files
--------------------

results_dir/
    test_ground_truth.npy
    cfo_predictions.npy
    picfo_predictions.npy

Outputs
-------

results_dir/vectorfield_physics/
    vectorfield_residual_vs_time.csv
    vectorfield_residual_on_rollout.png
    vectorfield_residual_on_ground_truth.png
    vectorfield_summary.txt
"""

import argparse
import csv
import os
from pathlib import Path
import sys


# ================================================================
# QUIET OPTIONAL LIBRARY OUTPUT
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

from utils.physics_swe import swe_residual


# ================================================================
# CONSTANTS
# ================================================================

EPS = 1e-12

CHANNEL_NAMES = (
    "h",
    "hu",
    "hv",
)


# ================================================================
# COMMAND-LINE ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate training-consistent vector-field "
            "SWE residuals for CFO and PI-CFO."
        )
    )

    # ------------------------------------------------------------
    # Existing rollout results
    # ------------------------------------------------------------

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Directory containing test_ground_truth.npy, "
            "cfo_predictions.npy, and picfo_predictions.npy."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    # ------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="FNO2d",
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
    # ------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cfo-ckpt-prefix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--cfo-ckpt-step",
        type=int,
        required=True,
    )

    # ------------------------------------------------------------
    # PI-CFO checkpoint
    # ------------------------------------------------------------

    parser.add_argument(
        "--picfo-ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--picfo-ckpt-prefix",
        type=str,
        default="",
    )

    parser.add_argument(
        "--picfo-ckpt-step",
        type=int,
        required=True,
    )

    # ------------------------------------------------------------
    # Physics
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
    # Optimizer parameters only needed to recreate TrainState
    # structure before Orbax restoration.
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

    return parser.parse_args()


# ================================================================
# RESTORE CFO
# ================================================================

def restore_cfo(
    *,
    input_shape,
    args,
):
    """Restore ordinary CFO."""

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


# ================================================================
# RESTORE PI-CFO
# ================================================================

def restore_picfo(
    *,
    input_shape,
    args,
):
    """Restore physics-informed CFO."""

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
# NUMERICAL TIME DERIVATIVE
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


# ================================================================
# GROUND-TRUTH NUMERICAL RESIDUAL
# ================================================================

def numerical_residual_vs_time(
    *,
    trajectory,
    times,
    dx,
    dy,
    gravity,
):
    """
    Compute finite-difference SWE residual.

    This is the same type of reference used by evaluate_picfo.py.
    """

    q_t = temporal_derivative(
        trajectory,
        times,
    )

    overall = []

    for time_index in range(
        trajectory.shape[1]
    ):

        q_now = jnp.asarray(
            trajectory[
                :,
                time_index,
            ]
        )

        qt_now = jnp.asarray(
            q_t[
                :,
                time_index,
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

        overall.append(
            float(
                np.mean(
                    residual**2
                )
            )
        )

    return np.asarray(
        overall,
        dtype=np.float64,
    )


# ================================================================
# DIRECT MODEL VECTOR FIELD
# ================================================================

def model_q_t(
    method,
    state,
    q: np.ndarray,
    time_value: float,
):
    """
    Evaluate the learned vector field

        q_t = N_theta(t, q).

    IMPORTANT
    ---------
    The FNO implementation expects time to have shape

        (batch,)

    rather than

        (batch, 1).

    Using (batch, 1) causes the time embedding to broadcast against
    the spatial tensor and creates an erroneous extra batch dimension:

        correct:
            (B, nx, ny, features)

        wrong:
            (B, B, nx, ny, features)
    """

    q_jax = jnp.asarray(
        q
    )

    batch_size = int(
        q_jax.shape[0]
    )

    # ============================================================
    # IMPORTANT FIX
    #
    # OLD / WRONG:
    #
    # physical_time = jnp.full(
    #     (batch_size, 1),
    #     time_value,
    # )
    #
    # NEW / CORRECT:
    #
    # shape = (batch_size,)
    # ============================================================

    physical_time = jnp.full(
        (
            batch_size,
        ),
        float(
            time_value
        ),
        dtype=q_jax.dtype,
    )

    q_t = method._model_apply(
        state.params,
        q_jax,
        physical_time,
        None,
    )

    return q_t


# ================================================================
# VECTOR-FIELD SWE RESIDUAL
# ================================================================

def vectorfield_residual_vs_time(
    *,
    method,
    state,
    trajectory,
    times,
    dx,
    dy,
    gravity,
):
    """
    Evaluate

        R = N_theta(t,q) + div(F(q))

    at every time point.
    """

    overall = []

    mass = []

    xmomentum = []

    ymomentum = []

    for time_index, time_value in enumerate(
        times
    ):

        q_now = np.asarray(
            trajectory[
                :,
                time_index,
            ],
            dtype=np.float32,
        )

        q_t_nn = model_q_t(
            method,
            state,
            q_now,
            float(
                time_value
            ),
        )

        residual = swe_residual(
            q=jnp.asarray(
                q_now
            ),
            q_t=q_t_nn,
            dx=dx,
            dy=dy,
            g=gravity,
        )

        residual = np.asarray(
            residual
        )

        overall.append(
            float(
                np.mean(
                    residual**2
                )
            )
        )

        mass.append(
            float(
                np.mean(
                    residual[
                        ...,
                        0
                    ]**2
                )
            )
        )

        xmomentum.append(
            float(
                np.mean(
                    residual[
                        ...,
                        1
                    ]**2
                )
            )
        )

        ymomentum.append(
            float(
                np.mean(
                    residual[
                        ...,
                        2
                    ]**2
                )
            )
        )

    return {
        "overall":
            np.asarray(
                overall,
                dtype=np.float64,
            ),

        "mass":
            np.asarray(
                mass,
                dtype=np.float64,
            ),

        "xmomentum":
            np.asarray(
                xmomentum,
                dtype=np.float64,
            ),

        "ymomentum":
            np.asarray(
                ymomentum,
                dtype=np.float64,
            ),
    }


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    *,
    output_dir,
    times,
    gt_numerical,
    cfo_rollout,
    picfo_rollout,
    cfo_gt,
    picfo_gt,
):
    """Save complete residual-vs-time table."""

    output_path = (
        output_dir
        /
        "vectorfield_residual_vs_time.csv"
    )

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "time",

                "ground_truth_numerical_pde_mse",

                "cfo_vectorfield_on_cfo_rollout",
                "picfo_vectorfield_on_picfo_rollout",

                "cfo_vectorfield_on_ground_truth",
                "picfo_vectorfield_on_ground_truth",

                "cfo_mass_residual_on_rollout",
                "cfo_xmomentum_residual_on_rollout",
                "cfo_ymomentum_residual_on_rollout",

                "picfo_mass_residual_on_rollout",
                "picfo_xmomentum_residual_on_rollout",
                "picfo_ymomentum_residual_on_rollout",

                "cfo_mass_residual_on_ground_truth",
                "cfo_xmomentum_residual_on_ground_truth",
                "cfo_ymomentum_residual_on_ground_truth",

                "picfo_mass_residual_on_ground_truth",
                "picfo_xmomentum_residual_on_ground_truth",
                "picfo_ymomentum_residual_on_ground_truth",
            ]
        )

        for i, time_value in enumerate(
            times
        ):

            writer.writerow(
                [
                    float(
                        time_value
                    ),

                    float(
                        gt_numerical[
                            i
                        ]
                    ),

                    float(
                        cfo_rollout[
                            "overall"
                        ][i]
                    ),

                    float(
                        picfo_rollout[
                            "overall"
                        ][i]
                    ),

                    float(
                        cfo_gt[
                            "overall"
                        ][i]
                    ),

                    float(
                        picfo_gt[
                            "overall"
                        ][i]
                    ),

                    float(
                        cfo_rollout[
                            "mass"
                        ][i]
                    ),

                    float(
                        cfo_rollout[
                            "xmomentum"
                        ][i]
                    ),

                    float(
                        cfo_rollout[
                            "ymomentum"
                        ][i]
                    ),

                    float(
                        picfo_rollout[
                            "mass"
                        ][i]
                    ),

                    float(
                        picfo_rollout[
                            "xmomentum"
                        ][i]
                    ),

                    float(
                        picfo_rollout[
                            "ymomentum"
                        ][i]
                    ),

                    float(
                        cfo_gt[
                            "mass"
                        ][i]
                    ),

                    float(
                        cfo_gt[
                            "xmomentum"
                        ][i]
                    ),

                    float(
                        cfo_gt[
                            "ymomentum"
                        ][i]
                    ),

                    float(
                        picfo_gt[
                            "mass"
                        ][i]
                    ),

                    float(
                        picfo_gt[
                            "xmomentum"
                        ][i]
                    ),

                    float(
                        picfo_gt[
                            "ymomentum"
                        ][i]
                    ),
                ]
            )


# ================================================================
# PLOT — OWN ROLLOUTS
# ================================================================

def plot_rollout_residual(
    *,
    output_dir,
    times,
    gt_numerical,
    cfo_rollout,
    picfo_rollout,
):
    """Plot vector-field residual on each model's own rollout."""

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.4,
        )
    )

    ax.plot(
        times,
        gt_numerical,
        label="Ground Truth — finite difference",
        linewidth=2,
    )

    ax.plot(
        times,
        cfo_rollout[
            "overall"
        ],
        label="CFO — vector field",
        linewidth=2,
    )

    ax.plot(
        times,
        picfo_rollout[
            "overall"
        ],
        label="PI-CFO — vector field",
        linewidth=2,
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Mean squared SWE residual"
    )

    ax.set_title(
        "Vector-Field SWE Residual on Model Rollouts"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "vectorfield_residual_on_rollout.png",
        dpi=300,
    )

    plt.close(
        fig
    )


# ================================================================
# PLOT — IDENTICAL GT STATES
# ================================================================

def plot_ground_truth_state_residual(
    *,
    output_dir,
    times,
    gt_numerical,
    cfo_gt,
    picfo_gt,
):
    """
    Compare both learned vector fields on exactly the same q_GT.
    """

    fig, ax = plt.subplots(
        figsize=(
            8.0,
            5.4,
        )
    )

    ax.plot(
        times,
        gt_numerical,
        label="Ground Truth — finite difference",
        linewidth=2,
    )

    ax.plot(
        times,
        cfo_gt[
            "overall"
        ],
        label="CFO vector field on GT state",
        linewidth=2,
    )

    ax.plot(
        times,
        picfo_gt[
            "overall"
        ],
        label="PI-CFO vector field on GT state",
        linewidth=2,
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Time"
    )

    ax.set_ylabel(
        "Mean squared SWE residual"
    )

    ax.set_title(
        "Vector-Field SWE Residual on Identical Ground-Truth States"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "vectorfield_residual_on_ground_truth.png",
        dpi=300,
    )

    plt.close(
        fig
    )


# ================================================================
# SUMMARY HELPERS
# ================================================================

def mean_value(
    data,
    key,
):
    """Mean of one residual component."""

    return float(
        np.mean(
            data[
                key
            ]
        )
    )


def percent_improvement(
    baseline,
    improved,
):
    """Positive number means improved is lower."""

    return (
        100.0
        *
        (
            baseline
            -
            improved
        )
        /
        (
            abs(
                baseline
            )
            +
            EPS
        )
    )


# ================================================================
# SAVE SUMMARY
# ================================================================

def save_summary(
    *,
    output_dir,
    gt_numerical,
    cfo_rollout,
    picfo_rollout,
    cfo_gt,
    picfo_gt,
):
    """Save human-readable diagnostic summary."""

    output_path = (
        output_dir
        /
        "vectorfield_summary.txt"
    )

    gt_mean = float(
        np.mean(
            gt_numerical
        )
    )

    cfo_rollout_mean = mean_value(
        cfo_rollout,
        "overall",
    )

    picfo_rollout_mean = mean_value(
        picfo_rollout,
        "overall",
    )

    cfo_gt_mean = mean_value(
        cfo_gt,
        "overall",
    )

    picfo_gt_mean = mean_value(
        picfo_gt,
        "overall",
    )

    rollout_improvement = percent_improvement(
        cfo_rollout_mean,
        picfo_rollout_mean,
    )

    gt_improvement = percent_improvement(
        cfo_gt_mean,
        picfo_gt_mean,
    )

    with output_path.open(
        "w"
    ) as f:

        f.write(
            "Training-Consistent Vector-Field Physics Evaluation\n"
        )

        f.write(
            "===================================================\n\n"
        )

        f.write(
            "Ground-truth finite-difference residual\n"
        )

        f.write(
            "---------------------------------------\n"
        )

        f.write(
            f"Mean PDE residual MSE: "
            f"{gt_mean:.8e}\n\n"
        )

        f.write(
            "A. Learned vector field evaluated on "
            "each model's own rollout\n"
        )

        f.write(
            "----------------------------------------------------\n"
        )

        f.write(
            f"CFO overall:     "
            f"{cfo_rollout_mean:.8e}\n"
        )

        f.write(
            f"PI-CFO overall:  "
            f"{picfo_rollout_mean:.8e}\n"
        )

        f.write(
            f"PI-CFO improvement: "
            f"{rollout_improvement:.4f}%\n\n"
        )

        f.write(
            "CFO components:\n"
        )

        f.write(
            f"  mass:       "
            f"{mean_value(cfo_rollout, 'mass'):.8e}\n"
        )

        f.write(
            f"  x-momentum: "
            f"{mean_value(cfo_rollout, 'xmomentum'):.8e}\n"
        )

        f.write(
            f"  y-momentum: "
            f"{mean_value(cfo_rollout, 'ymomentum'):.8e}\n\n"
        )

        f.write(
            "PI-CFO components:\n"
        )

        f.write(
            f"  mass:       "
            f"{mean_value(picfo_rollout, 'mass'):.8e}\n"
        )

        f.write(
            f"  x-momentum: "
            f"{mean_value(picfo_rollout, 'xmomentum'):.8e}\n"
        )

        f.write(
            f"  y-momentum: "
            f"{mean_value(picfo_rollout, 'ymomentum'):.8e}\n\n"
        )

        f.write(
            "B. Learned vector fields evaluated on the "
            "same ground-truth states\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            f"CFO overall:     "
            f"{cfo_gt_mean:.8e}\n"
        )

        f.write(
            f"PI-CFO overall:  "
            f"{picfo_gt_mean:.8e}\n"
        )

        f.write(
            f"PI-CFO improvement: "
            f"{gt_improvement:.4f}%\n\n"
        )

        f.write(
            "CFO components on GT states:\n"
        )

        f.write(
            f"  mass:       "
            f"{mean_value(cfo_gt, 'mass'):.8e}\n"
        )

        f.write(
            f"  x-momentum: "
            f"{mean_value(cfo_gt, 'xmomentum'):.8e}\n"
        )

        f.write(
            f"  y-momentum: "
            f"{mean_value(cfo_gt, 'ymomentum'):.8e}\n\n"
        )

        f.write(
            "PI-CFO components on GT states:\n"
        )

        f.write(
            f"  mass:       "
            f"{mean_value(picfo_gt, 'mass'):.8e}\n"
        )

        f.write(
            f"  x-momentum: "
            f"{mean_value(picfo_gt, 'xmomentum'):.8e}\n"
        )

        f.write(
            f"  y-momentum: "
            f"{mean_value(picfo_gt, 'ymomentum'):.8e}\n"
        )


# ================================================================
# MAIN
# ================================================================

def main():
    """Run vector-field physics evaluation."""

    args = parse_args()

    # ------------------------------------------------------------
    # Input directory
    # ------------------------------------------------------------

    results_dir = Path(
        args.results_dir
    ).resolve()

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist:\n"
            f"{results_dir}"
        )

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------

    if args.output_dir is None:

        output_dir = (
            results_dir
            /
            "vectorfield_physics"
        )

    else:

        output_dir = Path(
            args.output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Required arrays
    # ------------------------------------------------------------

    target_path = (
        results_dir
        /
        "test_ground_truth.npy"
    )

    cfo_path = (
        results_dir
        /
        "cfo_predictions.npy"
    )

    picfo_path = (
        results_dir
        /
        "picfo_predictions.npy"
    )

    for required_path in [
        target_path,
        cfo_path,
        picfo_path,
    ]:

        if not required_path.exists():
            raise FileNotFoundError(
                "Required result array not found:\n"
                f"{required_path}"
            )

    # ------------------------------------------------------------
    # Load arrays
    # ------------------------------------------------------------

    target = np.asarray(
        np.load(
            target_path
        ),
        dtype=np.float32,
    )

    cfo_pred = np.asarray(
        np.load(
            cfo_path
        ),
        dtype=np.float32,
    )

    picfo_pred = np.asarray(
        np.load(
            picfo_path
        ),
        dtype=np.float32,
    )

    if target.shape != cfo_pred.shape:

        raise ValueError(
            "CFO prediction shape does not match target.\n"
            f"Target: {target.shape}\n"
            f"CFO:    {cfo_pred.shape}"
        )

    if target.shape != picfo_pred.shape:

        raise ValueError(
            "PI-CFO prediction shape does not match target.\n"
            f"Target: {target.shape}\n"
            f"PI-CFO: {picfo_pred.shape}"
        )

    if target.ndim != 5:

        raise ValueError(
            "Expected shape "
            "(batch, time, nx, ny, channels), "
            f"received {target.shape}."
        )

    if target.shape[-1] != 3:

        raise ValueError(
            "Expected q=[h,hu,hv], "
            f"received {target.shape[-1]} channels."
        )

    input_shape = tuple(
        target.shape[
            2:
        ]
    )

    times = np.linspace(
        0.0,
        1.0,
        target.shape[
            1
        ],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "VECTOR-FIELD PHYSICS EVALUATION"
    )

    print(
        "============================================================"
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
        f"Architecture: "
        f"{args.model}"
    )

    print(
        f"Output directory:\n"
        f"{output_dir}"
    )

    # ------------------------------------------------------------
    # Restore CFO
    # ------------------------------------------------------------

    print(
        "\nRestoring CFO..."
    )

    (
        cfo_method,
        cfo_state,
    ) = restore_cfo(
        input_shape=input_shape,
        args=args,
    )

    print(
        "CFO restored."
    )

    # ------------------------------------------------------------
    # Restore PI-CFO
    # ------------------------------------------------------------

    print(
        "\nRestoring PI-CFO..."
    )

    (
        picfo_method,
        picfo_state,
    ) = restore_picfo(
        input_shape=input_shape,
        args=args,
    )

    print(
        "PI-CFO restored."
    )

    # ------------------------------------------------------------
    # Numerical reference
    # ------------------------------------------------------------

    print(
        "\nComputing ground-truth numerical residual..."
    )

    gt_numerical = numerical_residual_vs_time(
        trajectory=target,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ------------------------------------------------------------
    # CFO vector field on its own rollout
    # ------------------------------------------------------------

    print(
        "Computing CFO vector-field residual "
        "on CFO rollout..."
    )

    cfo_rollout = vectorfield_residual_vs_time(
        method=cfo_method,
        state=cfo_state,
        trajectory=cfo_pred,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ------------------------------------------------------------
    # PI-CFO vector field on its own rollout
    # ------------------------------------------------------------

    print(
        "Computing PI-CFO vector-field residual "
        "on PI-CFO rollout..."
    )

    picfo_rollout = vectorfield_residual_vs_time(
        method=picfo_method,
        state=picfo_state,
        trajectory=picfo_pred,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ------------------------------------------------------------
    # CFO on exact GT states
    # ------------------------------------------------------------

    print(
        "Computing CFO vector field "
        "on ground-truth states..."
    )

    cfo_gt = vectorfield_residual_vs_time(
        method=cfo_method,
        state=cfo_state,
        trajectory=target,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ------------------------------------------------------------
    # PI-CFO on exact GT states
    # ------------------------------------------------------------

    print(
        "Computing PI-CFO vector field "
        "on ground-truth states..."
    )

    picfo_gt = vectorfield_residual_vs_time(
        method=picfo_method,
        state=picfo_state,
        trajectory=target,
        times=times,
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------

    save_csv(
        output_dir=output_dir,
        times=times,
        gt_numerical=gt_numerical,
        cfo_rollout=cfo_rollout,
        picfo_rollout=picfo_rollout,
        cfo_gt=cfo_gt,
        picfo_gt=picfo_gt,
    )

    plot_rollout_residual(
        output_dir=output_dir,
        times=times,
        gt_numerical=gt_numerical,
        cfo_rollout=cfo_rollout,
        picfo_rollout=picfo_rollout,
    )

    plot_ground_truth_state_residual(
        output_dir=output_dir,
        times=times,
        gt_numerical=gt_numerical,
        cfo_gt=cfo_gt,
        picfo_gt=picfo_gt,
    )

    save_summary(
        output_dir=output_dir,
        gt_numerical=gt_numerical,
        cfo_rollout=cfo_rollout,
        picfo_rollout=picfo_rollout,
        cfo_gt=cfo_gt,
        picfo_gt=picfo_gt,
    )

    # ------------------------------------------------------------
    # Done
    # ------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "VECTOR-FIELD EVALUATION COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        f"Results saved to:\n"
        f"{output_dir}"
    )

    print(
        "\nCreated files:"
    )

    print(
        "  vectorfield_residual_vs_time.csv"
    )

    print(
        "  vectorfield_residual_on_rollout.png"
    )

    print(
        "  vectorfield_residual_on_ground_truth.png"
    )

    print(
        "  vectorfield_summary.txt"
    )


if __name__ == "__main__":
    main()