"""
Validation comparison for the explicit bed-response loss.

Models
------
Original PI-CFO:
    lambda_PDE = 0.03
    lambda_bed = 0.00

Bed-regularized PI-CFO:
    lambda_bed = 0.03
    lambda_bed = 0.10
    lambda_bed = 0.30
    lambda_bed = 0.50
    lambda_bed = 0.70

IMPORTANT
---------
Model selection uses ONLY the validation split.

Metrics
-------
1. Validation Relative L2
2. Same-ground-truth SWE residual
3. Final mass drift
4. Direct bed-response MSE
5. Bed-response cosine similarity
6. Bed-response magnitude ratio

Magnitude ratio:
    1.0  = ideal
    <1.0 = instantaneous terrain response too weak
    >1.0 = instantaneous terrain response too strong
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path
import sys

import jax
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

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
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
            "swe_bathy_32_id.h5"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "lambda_bed_validation"
        ),
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
        "--batch-size",
        type=int,
        default=8,
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
# RELATIVE L2
# ================================================================

def relative_l2(
    truth,
    prediction,
):

    numerator = np.linalg.norm(
        (
            prediction
            -
            truth
        ).reshape(
            truth.shape[0],
            -1,
        ),
        axis=1,
    )

    denominator = np.linalg.norm(
        truth.reshape(
            truth.shape[0],
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
# MASS DRIFT
# ================================================================

def final_mass_drift(
    q,
    dx,
    dy,
):

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

    mass0 = mass[
        :,
        0:1
    ]

    drift = (
        np.abs(
            mass
            -
            mass0
        )
        /
        np.maximum(
            np.abs(
                mass0
            ),
            1e-12,
        )
    )

    return float(
        np.mean(
            drift[
                :,
                -1
            ]
        )
    )


# ================================================================
# BUILD MODEL
# ================================================================

def build_method(
    lambda_bed,
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
        BathymetryBedRegularizedPICFO(
            model=model,
            input_shape=input_shape,
            condition_shape=(
                condition_shape
            ),
            gamma=1e-5,
            spline_type="quintic",
            lambda_pde=0.03,
            lambda_bed=float(
                lambda_bed
            ),
            dx=args.dx,
            dy=args.dy,
            gravity=args.gravity,
        )
    )

    return method


# ================================================================
# RESTORE CHECKPOINT
# ================================================================

def restore_state(
    method,
    *,
    ckpt_dir,
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
            ckpt_dir
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )

    return state


# ================================================================
# DIRECT BED METRICS
# ================================================================

def direct_bed_metrics(
    method,
    state,
    q_true,
    bathymetry,
    time,
    *,
    args,
):

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

    squared_error = 0.0

    error_count = 0

    dot_sum = 0.0

    expected_norm_sq = 0.0

    predicted_norm_sq = 0.0

    pde_squared = 0.0

    pde_count = 0

    for start in range(
        0,
        B * T,
        args.batch_size,
    ):

        end = min(
            start
            +
            args.batch_size,
            B * T,
        )

        q = jnp.asarray(
            q_flat[
                start:end
            ]
        )

        b = jnp.asarray(
            b_flat[
                start:end
            ]
        )

        t = jnp.asarray(
            t_flat[
                start:end
            ]
        )

        # --------------------------------------------------------
        # Same q, same t, real terrain
        # --------------------------------------------------------

        q_t_bathy = (
            method._model_apply(
                state.params,
                q,
                t,
                b,
            )
        )

        # --------------------------------------------------------
        # Same q, same t, flat terrain
        # --------------------------------------------------------

        q_t_flat = (
            method._model_apply(
                state.params,
                q,
                t,
                jnp.zeros_like(
                    b
                ),
            )
        )

        delta_q_t = (
            q_t_bathy
            -
            q_t_flat
        )

        # --------------------------------------------------------
        # Expected discrete SWE bed source
        # --------------------------------------------------------

        (
            source_x,
            source_y,
        ) = well_balanced_bed_source(
            q,
            b,
            dx=args.dx,
            dy=args.dy,
            g=args.gravity,
        )

        expected = jnp.stack(
            [
                -source_x,
                -source_y,
            ],
            axis=-1,
        )

        predicted = (
            delta_q_t[
                :,
                1:-1,
                1:-1,
                1:3,
            ]
        )

        expected_np = np.asarray(
            expected,
            dtype=np.float64,
        )

        predicted_np = np.asarray(
            predicted,
            dtype=np.float64,
        )

        difference = (
            predicted_np
            -
            expected_np
        )

        squared_error += float(
            np.sum(
                difference**2
            )
        )

        error_count += int(
            difference.size
        )

        dot_sum += float(
            np.sum(
                predicted_np
                *
                expected_np
            )
        )

        expected_norm_sq += float(
            np.sum(
                expected_np**2
            )
        )

        predicted_norm_sq += float(
            np.sum(
                predicted_np**2
            )
        )

        # --------------------------------------------------------
        # Same-ground-truth full PDE residual
        # --------------------------------------------------------

        residual = swe_bathy_residual(
            q,
            q_t_bathy,
            b,
            dx=args.dx,
            dy=args.dy,
            g=args.gravity,
        )

        residual_np = np.asarray(
            residual,
            dtype=np.float64,
        )

        pde_squared += float(
            np.sum(
                residual_np**2
            )
        )

        pde_count += int(
            residual_np.size
        )

    bed_mse = (
        squared_error
        /
        max(
            error_count,
            1,
        )
    )

    cosine = (
        dot_sum
        /
        max(
            np.sqrt(
                expected_norm_sq
                *
                predicted_norm_sq
            ),
            1e-12,
        )
    )

    magnitude_ratio = (
        np.sqrt(
            predicted_norm_sq
        )
        /
        max(
            np.sqrt(
                expected_norm_sq
            ),
            1e-12,
        )
    )

    pde_mse = (
        pde_squared
        /
        max(
            pde_count,
            1,
        )
    )

    return {
        "bed_response_mse":
            float(
                bed_mse
            ),

        "bed_response_cosine":
            float(
                cosine
            ),

        "bed_magnitude_ratio":
            float(
                magnitude_ratio
            ),

        "same_gt_pde_mse":
            float(
                pde_mse
            ),
    }


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
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ================================================================
# SUMMARY PLOTS
# ================================================================

def make_summary_plots(
    rows,
    output_dir,
):

    lambda_values = np.asarray(
        [
            row[
                "lambda_bed"
            ]
            for row
            in rows
        ],
        dtype=float,
    )

    validation_l2 = np.asarray(
        [
            row[
                "validation_rel_l2"
            ]
            for row
            in rows
        ]
    )

    bed_mse = np.asarray(
        [
            row[
                "bed_response_mse"
            ]
            for row
            in rows
        ]
    )

    cosine = np.asarray(
        [
            row[
                "bed_response_cosine"
            ]
            for row
            in rows
        ]
    )

    magnitude = np.asarray(
        [
            row[
                "bed_magnitude_ratio"
            ]
            for row
            in rows
        ]
    )

    # ------------------------------------------------------------
    # Validation error
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7,
            4.5,
        )
    )

    plt.plot(
        lambda_values,
        validation_l2,
        marker="o",
    )

    plt.xlabel(
        "lambda_bed"
    )

    plt.ylabel(
        "Validation Relative L2"
    )

    plt.title(
        "Trajectory accuracy vs bed-loss strength"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        output_dir
        /
        "01_validation_l2_vs_lambda_bed.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Bed MSE
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7,
            4.5,
        )
    )

    plt.plot(
        lambda_values,
        bed_mse,
        marker="o",
    )

    plt.xlabel(
        "lambda_bed"
    )

    plt.ylabel(
        "Bed-response MSE"
    )

    plt.title(
        "Direct bed-response error"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        output_dir
        /
        "02_bed_mse_vs_lambda_bed.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Cosine
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7,
            4.5,
        )
    )

    plt.plot(
        lambda_values,
        cosine,
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
    )

    plt.xlabel(
        "lambda_bed"
    )

    plt.ylabel(
        "Bed-response cosine similarity"
    )

    plt.title(
        "Bed-response direction / spatial-pattern agreement"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    fig.savefig(
        output_dir
        /
        "03_bed_cosine_vs_lambda_bed.png",
        dpi=220,
    )

    plt.close(
        fig
    )

    # ------------------------------------------------------------
    # Magnitude ratio
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(
            7,
            4.5,
        )
    )

    plt.plot(
        lambda_values,
        magnitude,
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Ideal magnitude",
    )

    plt.xlabel(
        "lambda_bed"
    )

    plt.ylabel(
        "Predicted / expected bed-response magnitude"
    )

    plt.title(
        "Bed-response magnitude calibration"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    fig.savefig(
        output_dir
        /
        "04_bed_magnitude_ratio_vs_lambda_bed.png",
        dpi=220,
    )

    plt.close(
        fig
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

    dataset = load_bathymetry_dataset(
        dataset_path
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

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    configurations = [
        {
            "name":
                "PI lambda_bed=0",

            "lambda_bed":
                0.00,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_pi_lam003/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_pi_cfo",
        },

        {
            "name":
                "Bed lambda=0.03",

            "lambda_bed":
                0.03,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_bed_lam003/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_bed_pi",
        },

        {
            "name":
                "Bed lambda=0.10",

            "lambda_bed":
                0.10,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_bed_lam01/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_bed_pi",
        },

        {
            "name":
                "Bed lambda=0.30",

            "lambda_bed":
                0.30,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_bed_lam03/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_bed_pi",
        },

        {
            "name":
                "Bed lambda=0.50",

            "lambda_bed":
                0.50,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_bed_lam05/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_bed_pi",
        },

        {
            "name":
                "Bed lambda=0.70",

            "lambda_bed":
                0.70,

            "ckpt":
                resolve_path(
                    "checkpoints/"
                    "bathy_bed_lam07/"
                    "seed0/"
                    "best"
                ),

            "prefix":
                "bathy_bed_pi",
        },
    ]

    rows = []

    # ============================================================
    # EVALUATE
    # ============================================================

    for config in configurations:

        print(
            "\n========================================"
        )

        print(
            config[
                "name"
            ]
        )

        print(
            "========================================"
        )

        method = build_method(
            config[
                "lambda_bed"
            ],
            input_shape,
            condition_shape,
            args,
        )

        state = restore_state(
            method,
            ckpt_dir=(
                config[
                    "ckpt"
                ]
            ),
            prefix=(
                config[
                    "prefix"
                ]
            ),
        )

        prediction = method.uniform_inference(
            state,
            q_true[
                :,
                0,
            ],
            trajectory_points_num=(
                q_true.shape[
                    1
                ]
            ),
            steps_per_segment=2,
            condition=bathymetry,
            method="RK4",
        )

        prediction = np.asarray(
            prediction,
            dtype=np.float32,
        )

        validation_rel_l2 = relative_l2(
            q_true,
            prediction,
        )

        mass_drift = final_mass_drift(
            prediction,
            args.dx,
            args.dy,
        )

        direct = direct_bed_metrics(
            method,
            state,
            q_true,
            bathymetry,
            time,
            args=args,
        )

        row = {
            "model":
                config[
                    "name"
                ],

            "lambda_bed":
                float(
                    config[
                        "lambda_bed"
                    ]
                ),

            "validation_rel_l2":
                float(
                    validation_rel_l2
                ),

            "same_gt_pde_mse":
                direct[
                    "same_gt_pde_mse"
                ],

            "final_mass_drift":
                float(
                    mass_drift
                ),

            "bed_response_mse":
                direct[
                    "bed_response_mse"
                ],

            "bed_response_cosine":
                direct[
                    "bed_response_cosine"
                ],

            "bed_magnitude_ratio":
                direct[
                    "bed_magnitude_ratio"
                ],
        }

        rows.append(
            row
        )

        print(
            f"Validation Rel L2: "
            f"{row['validation_rel_l2']:.6f}"
        )

        print(
            f"Same-GT PDE:       "
            f"{row['same_gt_pde_mse']:.6e}"
        )

        print(
            f"Final mass drift:  "
            f"{row['final_mass_drift']:.6e}"
        )

        print(
            f"Bed-response MSE:  "
            f"{row['bed_response_mse']:.6e}"
        )

        print(
            f"Bed cosine:        "
            f"{row['bed_response_cosine']:.6f}"
        )

        print(
            f"Bed magnitude:     "
            f"{row['bed_magnitude_ratio']:.6f}"
        )

        del prediction
        del state
        del method

        gc.collect()

        try:
            jax.clear_caches()
        except AttributeError:
            pass

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = (
        output_dir
        /
        "lambda_bed_comparison.csv"
    )

    save_csv(
        rows,
        csv_path,
    )

    make_summary_plots(
        rows,
        output_dir,
    )

    # ============================================================
    # TABLE
    # ============================================================

    print(
        "\n"
        +
        "=" * 116
    )

    print(
        "LAMBDA_BED VALIDATION COMPARISON"
    )

    print(
        "=" * 116
    )

    print(
        f"{'Model':<21}"
        f"{'Rel L2':>12}"
        f"{'PDE MSE':>15}"
        f"{'Mass':>13}"
        f"{'Bed MSE':>15}"
        f"{'Cosine':>12}"
        f"{'Mag ratio':>13}"
    )

    print(
        "-" * 116
    )

    for row in rows:

        print(
            f"{row['model']:<21}"
            f"{row['validation_rel_l2']:>12.6f}"
            f"{row['same_gt_pde_mse']:>15.4e}"
            f"{row['final_mass_drift']:>13.4e}"
            f"{row['bed_response_mse']:>15.4e}"
            f"{row['bed_response_cosine']:>12.4f}"
            f"{row['bed_magnitude_ratio']:>13.4f}"
        )

    print(
        "=" * 116
    )

    print(
        "\nIdeal bed magnitude ratio = 1.0"
    )

    print(
        f"\nSaved results to:\n"
        f"{output_dir}"
    )


if __name__ == "__main__":

    main()