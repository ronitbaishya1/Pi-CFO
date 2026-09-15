"""
Train Bathymetry PI-CFO with explicit bed-response regularization.

Objective:

    L
    =
    L_CFO
    +
    lambda_PDE * L_PDE
    +
    lambda_bed * L_bed

Recommended first experiment:

    lambda_PDE = 0.03
    lambda_bed = 0.10

Do not overwrite the existing Bathy-CFO or Bathy-PI-CFO runs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

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
# ROOT
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
# IMPORTS
# ================================================================

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from models.factory import (
    build_model,
)

from train import (
    CFOTrainArgs,
)

from train_bathy_bed import (
    train_bathy_bed_picfo,
)

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.bathy_training import (
    broadcast_time_for_trajectories,
    build_conditioned_spline_dataloader,
)

from utils.data import (
    load_partial_data,
)

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)

from utils.seed import (
    set_global_seed,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
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

    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--lambda-bed",
        type=float,
        default=0.10,
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
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
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
        "--spline-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--partial-train-ratio",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--partial-data-seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_bed_pi"
        ),
    )

    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="bathy_bed_pi",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/"
            "bathy_bed_pi"
        ),
    )

    parser.add_argument(
        "--running-ckpt-interval",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


# ================================================================
# PATH HELPER
# ================================================================

def resolve_path(
    path_string,
):

    path = Path(
        path_string
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    set_global_seed(
        args.seed
    )

    dataset_path = resolve_path(
        args.dataset_path
    )

    # ============================================================
    # DATA
    # ============================================================

    dataset = load_bathymetry_dataset(
        dataset_path
    )

    train_split = require_split(
        dataset,
        "train",
    )

    eval_split = require_split(
        dataset,
        "eval",
    )

    test_split = require_split(
        dataset,
        "test_id",
    )

    train_data = np.asarray(
        train_split[
            "q"
        ],
        dtype=np.float32,
    )

    train_bathy = np.asarray(
        train_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    eval_data = np.asarray(
        eval_split[
            "q"
        ],
        dtype=np.float32,
    )

    eval_bathy = np.asarray(
        eval_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    test_data = np.asarray(
        test_split[
            "q"
        ],
        dtype=np.float32,
    )

    test_bathy = np.asarray(
        test_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    full_train_time = (
        broadcast_time_for_trajectories(
            train_split[
                "time"
            ],
            train_data.shape[
                0
            ],
            train_data.shape[
                1
            ],
            dtype=train_data.dtype,
        )
    )

    # ============================================================
    # OPTIONAL TEMPORAL SUBSAMPLING
    # ============================================================

    if (
        args.partial_train_ratio
        <
        1.0
    ):

        (
            train_data,
            train_time,
        ) = load_partial_data(
            train_data,
            ratio=(
                args.partial_train_ratio
            ),
            seed=(
                args.partial_data_seed
            ),
        )

    else:

        train_time = (
            full_train_time
        )

    input_shape = tuple(
        train_data.shape[
            2:
        ]
    )

    condition_shape = tuple(
        train_bathy.shape[
            1:
        ]
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    method = BathymetryBedRegularizedPICFO(
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
        lambda_bed=(
            args.lambda_bed
        ),
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )

    # ============================================================
    # SUMMARY
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "BATHYMETRY BED-REGULARIZED PI-CFO"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"State shape: "
        f"{input_shape}"
    )

    print(
        f"Bathymetry shape: "
        f"{condition_shape}"
    )

    print(
        f"Train/eval/test: "
        f"{len(train_data)}/"
        f"{len(eval_data)}/"
        f"{len(test_data)}"
    )

    print(
        f"Snapshots retained: "
        f"{train_data.shape[1]}"
    )

    print(
        f"lambda_PDE: "
        f"{args.lambda_pde}"
    )

    print(
        f"lambda_bed: "
        f"{args.lambda_bed}"
    )

    print(
        "\nObjective:"
    )

    print(
        "L = L_CFO "
        "+ lambda_PDE*L_PDE "
        "+ lambda_bed*L_bed"
    )

    print(
        "========================================\n"
    )

    if args.dry_run:

        return

    # ============================================================
    # SPLINE DATA
    # ============================================================

    print(
        "Preparing spline dataloader..."
    )

    spline_loader = (
        build_conditioned_spline_dataloader(
            train_data=train_data,
            bathymetry=train_bathy,
            time=train_time,
            spline_type=(
                args.spline_type
            ),
            batch_size=(
                args.batch_size
            ),
            spline_batch_size=(
                args.spline_batch_size
            ),
            epochs=(
                args.epochs
            ),
            seed=(
                args.seed
            ),
        )
    )

    # ============================================================
    # CHECKPOINT PATH
    # ============================================================

    ckpt_root = resolve_path(
        args.ckpt_dir
    )

    running_interval = (
        args.eval_interval
        if (
            args.running_ckpt_interval
            is None
        )
        else
        args.running_ckpt_interval
    )

    train_args = CFOTrainArgs(
        num_epochs=(
            args.epochs
        ),
        random_seed=(
            args.seed
        ),
        use_wandb=(
            args.use_wandb
        ),
        learning_rate=(
            args.lr
        ),
        beta1=(
            args.beta1
        ),
        beta2=(
            args.beta2
        ),
        do_eval=(
            not args.no_eval
        ),
        eval_interval=(
            args.eval_interval
        ),
        irregular_time=False,

        running_ckpt_dir=str(
            ckpt_root
            /
            "running"
        ),

        running_ckpt_prefix=(
            args.ckpt_prefix
        ),

        running_ckpt_interval=(
            running_interval
        ),

        running_ckpt_max_to_keep=3,

        best_ckpt_dir=str(
            ckpt_root
            /
            "best"
        ),

        best_ckpt_prefix=(
            args.ckpt_prefix
        ),
    )

    eval_dataset = (
        eval_data[
            :,
            0,
        ],
        eval_data,
        eval_bathy,
    )

    # ============================================================
    # TRAIN
    # ============================================================

    output = train_bathy_bed_picfo(
        method,
        spline_loader,
        train_args,
        eval_dataset=eval_dataset,
    )

    # ============================================================
    # BEST STATE
    # ============================================================

    state_for_test = output[
        "state"
    ]

    if np.isfinite(
        output[
            "best_l2_error"
        ]
    ):

        state_for_test = output[
            "best_state"
        ]

    # ============================================================
    # TEST ROLLOUT
    # ============================================================

    print(
        "\nRunning final ID test inference..."
    )

    test_pred = method.uniform_inference(
        state_for_test,
        test_data[
            :,
            0,
        ],
        trajectory_points_num=(
            test_data.shape[
                1
            ]
        ),
        steps_per_segment=2,
        condition=test_bathy,
        method="RK4",
    )

    test_pred = np.asarray(
        test_pred
    )

    test_rel_l2 = float(
        relative_L2_error(
            test_data,
            test_pred,
        )
    )

    test_rmse = float(
        rmse(
            test_data,
            test_pred,
        )
    )

    test_rel_fro = float(
        relative_frobenius_error(
            test_data,
            test_pred,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL BED-PI-CFO ID TEST"
    )

    print(
        "========================================"
    )

    print(
        "Relative L2: "
        f"{test_rel_l2:.6f}"
    )

    print(
        "RMSE:        "
        f"{test_rmse:.6f}"
    )

    print(
        "Rel Fro:     "
        f"{test_rel_fro:.6f}"
    )

    print(
        "Best validation epoch: "
        f"{output['best_epoch']}"
    )

    print(
        "Best validation Rel L2: "
        f"{output['best_l2_error']:.6f}"
    )

    print(
        "========================================"
    )

    # ============================================================
    # SAVE
    # ============================================================

    lambda_bed_label = (
        f"{args.lambda_bed:g}"
        .replace(
            ".",
            "p",
        )
    )

    run_dir = (
        resolve_path(
            args.results_dir
        )
        /
        f"lambda_bed_{lambda_bed_label}"
        /
        f"seed{args.seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        run_dir
        /
        "test_true.npy",
        test_data,
    )

    np.save(
        run_dir
        /
        "test_pred.npy",
        test_pred,
    )

    np.save(
        run_dir
        /
        "test_bathymetry.npy",
        test_bathy,
    )

    np.savez(
        run_dir
        /
        "metrics.npz",

        relative_l2=(
            test_rel_l2
        ),

        rmse=(
            test_rmse
        ),

        relative_frobenius=(
            test_rel_fro
        ),

        best_eval_relative_l2=float(
            output[
                "best_l2_error"
            ]
        ),

        best_epoch=int(
            output[
                "best_epoch"
            ]
        ),

        lambda_pde=float(
            args.lambda_pde
        ),

        lambda_bed=float(
            args.lambda_bed
        ),

        final_total_loss=float(
            output[
                "total_loss_log"
            ][
                -1
            ]
        ),

        final_cfo_loss=float(
            output[
                "cfo_loss_log"
            ][
                -1
            ]
        ),

        final_pde_loss=float(
            output[
                "pde_loss_log"
            ][
                -1
            ]
        ),

        final_bed_loss=float(
            output[
                "bed_loss_log"
            ][
                -1
            ]
        ),

        final_weighted_pde=float(
            output[
                "weighted_pde_loss_log"
            ][
                -1
            ]
        ),

        final_weighted_bed=float(
            output[
                "weighted_bed_loss_log"
            ][
                -1
            ]
        ),
    )

    print(
        f"\nSaved results to:\n"
        f"{run_dir}"
    )


if __name__ == "__main__":

    main()