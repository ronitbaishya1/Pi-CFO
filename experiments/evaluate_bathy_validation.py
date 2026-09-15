"""
Phase 55: validation-set comparison for bathymetry-conditioned CFO models.

This script compares:

    Bathy-CFO
    Bathy-PI-CFO, lambda_PDE = 0.03
    Bathy-PI-CFO, lambda_PDE = 0.10
    Bathy-PI-CFO, lambda_PDE = 0.30

IMPORTANT
---------
This script uses ONLY the validation split.

The test_id split is deliberately not used for lambda selection.

Metrics
-------
1. Validation Relative L2 trajectory error

2. Rollout SWE residual MSE
   - q is the model's own rollout
   - q_t is estimated from the rollout using finite differences in time
   - evaluates whether the generated trajectory approximately satisfies
     the variable-bottom shallow-water equations

3. Same-GT vector-field SWE residual MSE
   - q is the SAME ground-truth validation state for every model
   - q_t = N_theta(t, q, b)
   - this isolates the physical consistency of the learned vector field

4. Mean relative mass drift

5. Final relative mass drift

Ground-truth finite-difference residual and mass drift are also printed
as numerical reference values.

Run from the project root:

    python experiments/evaluate_bathy_validation.py
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
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

from utils.metrics import (
    relative_L2_error,
    rmse,
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
            "Phase 55 validation comparison "
            "for bathymetry CFO models."
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
        "--pi003-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_pi_lam003/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--pi01-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_pi_lam01/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--pi03-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_pi_lam03/"
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
        "--pi-prefix",
        type=str,
        default="bathy_pi_cfo",
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
        "--gamma",
        type=float,
        default=1e-5,
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
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--vector-field-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/"
            "bathy_validation_comparison"
        ),
    )

    return parser.parse_args()


# ================================================================
# PATH HELPER
# ================================================================

def resolve_project_path(
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
# BUILD METHOD
# ================================================================

def build_method(
    *,
    input_shape,
    condition_shape,
    gamma: float,
    spline_type: str,
    dx: float,
    dy: float,
    gravity: float,
    lambda_pde: float | None,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    # ------------------------------------------------------------
    # STANDARD BATHY-CFO
    # ------------------------------------------------------------

    if lambda_pde is None:

        method = ContinuousFlowOperator(
            model=model,
            input_shape=input_shape,
            gamma=float(
                gamma
            ),
            spline_type=(
                spline_type
            ),
            use_condition=True,
            condition_shape=(
                condition_shape
            ),
        )

        return method

    # ------------------------------------------------------------
    # BATHY-PI-CFO
    # ------------------------------------------------------------

    method = (
        BathymetryPhysicsInformedCFO(
            model=model,
            input_shape=input_shape,
            condition_shape=(
                condition_shape
            ),
            gamma=float(
                gamma
            ),
            spline_type=(
                spline_type
            ),
            lambda_pde=float(
                lambda_pde
            ),
            dx=float(
                dx
            ),
            dy=float(
                dy
            ),
            gravity=float(
                gravity
            ),
        )
    )

    return method


# ================================================================
# RESTORE MODEL
# ================================================================

def restore_model(
    *,
    label: str,
    ckpt_dir: Path,
    prefix: str,
    input_shape,
    condition_shape,
    gamma: float,
    spline_type: str,
    dx: float,
    dy: float,
    gravity: float,
    lambda_pde: float | None,
    seed: int,
):

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        f"RESTORING: {label}"
    )

    print(
        "=" * 72
    )

    print(
        f"checkpoint dir: {ckpt_dir}"
    )

    print(
        f"prefix:         {prefix}"
    )

    method = build_method(
        input_shape=input_shape,
        condition_shape=condition_shape,
        gamma=gamma,
        spline_type=spline_type,
        dx=dx,
        dy=dy,
        gravity=gravity,
        lambda_pde=lambda_pde,
    )

    target_state = init_cfo_train_state(
        method,
        seed=seed,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
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

    print(
        "checkpoint restore: OK"
    )

    return (
        method,
        state,
    )


# ================================================================
# RELATIVE MASS DRIFT
# ================================================================

def mass_drift_metrics(
    q: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> tuple[
    float,
    float,
    float,
]:

    """
    Compute relative mass drift.

    M(t) = sum h dx dy

    Relative drift:

        |M(t)-M(0)| / |M(0)|

    Returns:
        mean drift excluding t=0
        final drift
        maximum drift
    """

    q = np.asarray(
        q,
        dtype=np.float64,
    )

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

    mass0 = mass[
        :,
        0:1
    ]

    denominator = np.maximum(
        np.abs(
            mass0
        ),
        1e-12,
    )

    relative_drift = (
        np.abs(
            mass
            -
            mass0
        )
        /
        denominator
    )

    if relative_drift.shape[1] > 1:

        mean_drift = float(
            np.mean(
                relative_drift[
                    :,
                    1:
                ]
            )
        )

    else:

        mean_drift = 0.0

    final_drift = float(
        np.mean(
            relative_drift[
                :,
                -1
            ]
        )
    )

    max_drift = float(
        np.max(
            relative_drift
        )
    )

    return (
        mean_drift,
        final_drift,
        max_drift,
    )


# ================================================================
# FINITE-DIFFERENCE TIME DERIVATIVE
# ================================================================

def finite_difference_time_derivative(
    q: np.ndarray,
    time: np.ndarray,
) -> np.ndarray:

    q = np.asarray(
        q,
        dtype=np.float32,
    )

    time = np.asarray(
        time,
        dtype=np.float64,
    )

    if time.ndim != 1:

        raise ValueError(
            "Validation time must have "
            f"shape (T,), got {time.shape}."
        )

    if q.shape[1] != len(
        time
    ):

        raise ValueError(
            "Time length does not match "
            "trajectory length."
        )

    if len(
        time
    ) >= 3:

        edge_order = 2

    else:

        edge_order = 1

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
# PHYSICS RESIDUAL MSE
# ================================================================

def physics_residual_mse(
    q: np.ndarray,
    q_t: np.ndarray,
    bathymetry: np.ndarray,
    *,
    dx: float,
    dy: float,
    gravity: float,
    batch_size: int = 64,
) -> float:

    """
    Evaluate the variable-bottom SWE residual over an entire
    collection of trajectories.

    q shape:
        (B,T,H,W,3)

    q_t shape:
        (B,T,H,W,3)

    bathymetry shape:
        (B,H,W,1)
    """

    q = np.asarray(
        q,
        dtype=np.float32,
    )

    q_t = np.asarray(
        q_t,
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float32,
    )

    B, T, H, W, C = (
        q.shape
    )

    if C != 3:

        raise ValueError(
            "Expected 3 SWE state channels."
        )

    if q_t.shape != q.shape:

        raise ValueError(
            "q and q_t must have "
            "the same shape."
        )

    if bathymetry.shape != (
        B,
        H,
        W,
        1,
    ):

        raise ValueError(
            "Unexpected bathymetry shape: "
            f"{bathymetry.shape}"
        )

    q_flat = q.reshape(
        B * T,
        H,
        W,
        C,
    )

    qt_flat = q_t.reshape(
        B * T,
        H,
        W,
        C,
    )

    bathy_flat = np.repeat(
        bathymetry,
        T,
        axis=0,
    )

    squared_sum = 0.0
    element_count = 0

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

        residual = (
            swe_bathy_residual(
                jnp.asarray(
                    q_flat[
                        start:end
                    ]
                ),
                jnp.asarray(
                    qt_flat[
                        start:end
                    ]
                ),
                jnp.asarray(
                    bathy_flat[
                        start:end
                    ]
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

        residual_np = np.asarray(
            residual,
            dtype=np.float64,
        )

        squared_sum += float(
            np.sum(
                residual_np
                **
                2
            )
        )

        element_count += int(
            residual_np.size
        )

    if element_count == 0:

        raise RuntimeError(
            "No residual elements evaluated."
        )

    return (
        squared_sum
        /
        element_count
    )


# ================================================================
# NEURAL VECTOR FIELD ON GROUND-TRUTH STATES
# ================================================================

def evaluate_vector_field_on_ground_truth(
    method,
    state,
    q_true: np.ndarray,
    bathymetry: np.ndarray,
    time: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:

    """
    Compute

        q_t^NN = N_theta(t, q_true, b)

    on the exact same validation states for every model.

    This is useful because trajectory drift is removed from the
    comparison.
    """

    q_true = np.asarray(
        q_true,
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float32,
    )

    time = np.asarray(
        time,
        dtype=np.float32,
    )

    B, T, H, W, C = (
        q_true.shape
    )

    q_flat = q_true.reshape(
        B * T,
        H,
        W,
        C,
    )

    # q_flat order is:
    #
    # trajectory 0: t0,t1,...,tT
    # trajectory 1: t0,t1,...,tT
    #
    # so time must be tiled and terrain repeated.

    time_flat = np.tile(
        time,
        B,
    )

    bathy_flat = np.repeat(
        bathymetry,
        T,
        axis=0,
    )

    predictions = []

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

        t_batch = jnp.asarray(
            time_flat[
                start:end
            ]
        )

        b_batch = jnp.asarray(
            bathy_flat[
                start:end
            ]
        )

        pred = method._model_apply(
            state.params,
            q_batch,
            t_batch,
            b_batch,
        )

        predictions.append(
            np.asarray(
                pred,
                dtype=np.float32,
            )
        )

    q_t_pred = np.concatenate(
        predictions,
        axis=0,
    )

    q_t_pred = q_t_pred.reshape(
        B,
        T,
        H,
        W,
        C,
    )

    return q_t_pred


# ================================================================
# EVALUATE ONE MODEL
# ================================================================

def evaluate_model(
    *,
    label: str,
    method,
    state,
    q_true: np.ndarray,
    bathymetry: np.ndarray,
    time: np.ndarray,
    dx: float,
    dy: float,
    gravity: float,
    steps_per_segment: int,
    vector_field_batch_size: int,
) -> dict:

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        f"EVALUATING: {label}"
    )

    print(
        "=" * 72
    )

    # ============================================================
    # 1. MODEL ROLLOUT
    # ============================================================

    print(
        "Generating validation rollout..."
    )

    q_pred = method.uniform_inference(
        state,
        q_true[
            :,
            0
        ],
        trajectory_points_num=(
            q_true.shape[
                1
            ]
        ),
        steps_per_segment=(
            steps_per_segment
        ),
        condition=bathymetry,
        method="RK4",
    )

    q_pred = np.asarray(
        q_pred,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(
            q_pred
        )
    ):

        raise RuntimeError(
            f"{label} rollout contains "
            "NaN or Inf."
        )

    # ============================================================
    # 2. TRAJECTORY ACCURACY
    # ============================================================

    rel_l2 = float(
        relative_L2_error(
            q_true,
            q_pred,
        )
    )

    rmse_value = float(
        rmse(
            q_true,
            q_pred,
        )
    )

    print(
        f"Validation Relative L2: "
        f"{rel_l2:.8f}"
    )

    # ============================================================
    # 3. ROLLOUT PDE RESIDUAL
    #
    # q_t estimated from generated trajectory.
    # ============================================================

    print(
        "Computing rollout SWE residual..."
    )

    q_t_rollout = (
        finite_difference_time_derivative(
            q_pred,
            time,
        )
    )

    rollout_pde = (
        physics_residual_mse(
            q_pred,
            q_t_rollout,
            bathymetry,
            dx=dx,
            dy=dy,
            gravity=gravity,
        )
    )

    print(
        "Rollout SWE residual MSE: "
        f"{rollout_pde:.8e}"
    )

    # ============================================================
    # 4. SAME-GROUND-TRUTH VECTOR-FIELD DIAGNOSTIC
    #
    # Every model sees exactly the same q_true and b.
    # ============================================================

    print(
        "Evaluating learned vector field "
        "on ground-truth states..."
    )

    q_t_vector_field = (
        evaluate_vector_field_on_ground_truth(
            method,
            state,
            q_true,
            bathymetry,
            time,
            batch_size=(
                vector_field_batch_size
            ),
        )
    )

    same_gt_pde = (
        physics_residual_mse(
            q_true,
            q_t_vector_field,
            bathymetry,
            dx=dx,
            dy=dy,
            gravity=gravity,
        )
    )

    print(
        "Same-GT vector-field SWE residual MSE: "
        f"{same_gt_pde:.8e}"
    )

    # ============================================================
    # 5. MASS DRIFT
    # ============================================================

    (
        mean_mass_drift,
        final_mass_drift,
        max_mass_drift,
    ) = mass_drift_metrics(
        q_pred,
        dx=dx,
        dy=dy,
    )

    print(
        "Mean relative mass drift: "
        f"{mean_mass_drift:.8e}"
    )

    print(
        "Final relative mass drift: "
        f"{final_mass_drift:.8e}"
    )

    print(
        "Maximum relative mass drift: "
        f"{max_mass_drift:.8e}"
    )

    return {
        "model":
            label,

        "relative_l2":
            rel_l2,

        "rmse":
            rmse_value,

        "rollout_pde_mse":
            float(
                rollout_pde
            ),

        "same_gt_vector_field_pde_mse":
            float(
                same_gt_pde
            ),

        "mean_mass_drift":
            float(
                mean_mass_drift
            ),

        "final_mass_drift":
            float(
                final_mass_drift
            ),

        "max_mass_drift":
            float(
                max_mass_drift
            ),
    }


# ================================================================
# PRINT COMPARISON TABLE
# ================================================================

def print_comparison_table(
    rows: list[dict],
) -> None:

    print(
        "\n"
        +
        "=" * 126
    )

    print(
        "PHASE 55 — VALIDATION COMPARISON"
    )

    print(
        "=" * 126
    )

    header = (
        f"{'Model':<22}"
        f"{'Rel L2':>13}"
        f"{'Rollout PDE':>18}"
        f"{'Same-GT VF PDE':>18}"
        f"{'Mean Mass':>18}"
        f"{'Final Mass':>18}"
    )

    print(
        header
    )

    print(
        "-" * 126
    )

    for row in rows:

        print(
            f"{row['model']:<22}"
            f"{row['relative_l2']:>13.6f}"
            f"{row['rollout_pde_mse']:>18.6e}"
            f"{row['same_gt_vector_field_pde_mse']:>18.6e}"
            f"{row['mean_mass_drift']:>18.6e}"
            f"{row['final_mass_drift']:>18.6e}"
        )

    print(
        "=" * 126
    )


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    rows: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "relative_l2",
        "rmse",
        "rollout_pde_mse",
        "same_gt_vector_field_pde_mse",
        "mean_mass_drift",
        "final_mass_drift",
        "max_mass_drift",
    ]

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    # ============================================================
    # PATHS
    # ============================================================

    dataset_path = resolve_project_path(
        args.dataset_path
    )

    cfo_ckpt_dir = resolve_project_path(
        args.cfo_ckpt_dir
    )

    pi003_ckpt_dir = resolve_project_path(
        args.pi003_ckpt_dir
    )

    pi01_ckpt_dir = resolve_project_path(
        args.pi01_ckpt_dir
    )

    pi03_ckpt_dir = resolve_project_path(
        args.pi03_ckpt_dir
    )

    results_dir = resolve_project_path(
        args.results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # LOAD VALIDATION DATA
    # ============================================================

    print(
        "\nLoading bathymetry dataset..."
    )

    dataset = (
        load_bathymetry_dataset(
            dataset_path
        )
    )

    validation = require_split(
        dataset,
        "eval",
    )

    q_true = np.asarray(
        validation[
            "q"
        ],
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        validation[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    time = np.asarray(
        validation[
            "time"
        ],
        dtype=np.float32,
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
        "\n"
        +
        "=" * 72
    )

    print(
        "VALIDATION DATA"
    )

    print(
        "=" * 72
    )

    print(
        f"Dataset: "
        f"{dataset_path}"
    )

    print(
        f"q shape: "
        f"{q_true.shape}"
    )

    print(
        f"bathymetry shape: "
        f"{bathymetry.shape}"
    )

    print(
        f"time shape: "
        f"{time.shape}"
    )

    print(
        f"input shape: "
        f"{input_shape}"
    )

    print(
        f"condition shape: "
        f"{condition_shape}"
    )

    print(
        f"time range: "
        f"{float(time[0]):.4f} "
        f"-> "
        f"{float(time[-1]):.4f}"
    )

    print(
        "=" * 72
    )

    # ============================================================
    # GROUND-TRUTH NUMERICAL REFERENCE
    # ============================================================

    print(
        "\nComputing ground-truth "
        "finite-difference reference..."
    )

    q_t_true_fd = (
        finite_difference_time_derivative(
            q_true,
            time,
        )
    )

    ground_truth_pde = (
        physics_residual_mse(
            q_true,
            q_t_true_fd,
            bathymetry,
            dx=args.dx,
            dy=args.dy,
            gravity=args.gravity,
        )
    )

    (
        gt_mean_mass,
        gt_final_mass,
        gt_max_mass,
    ) = mass_drift_metrics(
        q_true,
        dx=args.dx,
        dy=args.dy,
    )

    print(
        "\nGROUND-TRUTH NUMERICAL REFERENCE"
    )

    print(
        "Finite-difference SWE residual MSE: "
        f"{ground_truth_pde:.8e}"
    )

    print(
        "Mean relative mass drift: "
        f"{gt_mean_mass:.8e}"
    )

    print(
        "Final relative mass drift: "
        f"{gt_final_mass:.8e}"
    )

    print(
        "Maximum relative mass drift: "
        f"{gt_max_mass:.8e}"
    )

    # ============================================================
    # MODEL CONFIGURATIONS
    # ============================================================

    configurations = [
        {
            "label":
                "Bathy-CFO",

            "lambda_pde":
                None,

            "ckpt_dir":
                cfo_ckpt_dir,

            "prefix":
                args.cfo_prefix,
        },

        {
            "label":
                "PI lambda=0.03",

            "lambda_pde":
                0.03,

            "ckpt_dir":
                pi003_ckpt_dir,

            "prefix":
                args.pi_prefix,
        },

        {
            "label":
                "PI lambda=0.10",

            "lambda_pde":
                0.10,

            "ckpt_dir":
                pi01_ckpt_dir,

            "prefix":
                args.pi_prefix,
        },

        {
            "label":
                "PI lambda=0.30",

            "lambda_pde":
                0.30,

            "ckpt_dir":
                pi03_ckpt_dir,

            "prefix":
                args.pi_prefix,
        },
    ]

    results = []

    # ============================================================
    # EVALUATE MODELS ONE AT A TIME
    # ============================================================

    for config in configurations:

        (
            method,
            state,
        ) = restore_model(
            label=config[
                "label"
            ],
            ckpt_dir=config[
                "ckpt_dir"
            ],
            prefix=config[
                "prefix"
            ],
            input_shape=input_shape,
            condition_shape=(
                condition_shape
            ),
            gamma=args.gamma,
            spline_type=(
                args.spline_type
            ),
            dx=args.dx,
            dy=args.dy,
            gravity=args.gravity,
            lambda_pde=config[
                "lambda_pde"
            ],
            seed=args.seed,
        )

        row = evaluate_model(
            label=config[
                "label"
            ],
            method=method,
            state=state,
            q_true=q_true,
            bathymetry=bathymetry,
            time=time,
            dx=args.dx,
            dy=args.dy,
            gravity=args.gravity,
            steps_per_segment=(
                args.steps_per_segment
            ),
            vector_field_batch_size=(
                args.vector_field_batch_size
            ),
        )

        results.append(
            row
        )

        # --------------------------------------------------------
        # Free memory before restoring next 9.5M parameter FNO.
        # --------------------------------------------------------

        del state
        del method

        gc.collect()

        try:

            jax.clear_caches()

        except AttributeError:

            pass

    # ============================================================
    # TABLE
    # ============================================================

    print_comparison_table(
        results
    )

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = (
        results_dir
        /
        "validation_metrics.csv"
    )

    save_csv(
        results,
        csv_path,
    )

    np.savez(
        results_dir
        /
        "ground_truth_reference.npz",

        finite_difference_pde_mse=float(
            ground_truth_pde
        ),

        mean_mass_drift=float(
            gt_mean_mass
        ),

        final_mass_drift=float(
            gt_final_mass
        ),

        max_mass_drift=float(
            gt_max_mass
        ),
    )

    print(
        "\nSaved validation comparison:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Use these VALIDATION results "
        "to choose lambda_PDE."
    )

    print(
        "Do not use test_id results "
        "for lambda selection."
    )


if __name__ == "__main__":

    main()