"""Phase 53: train bathymetry-conditioned CFO on 2D shallow-water data."""

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

from cfo import (
    ContinuousFlowOperator,
)

from models.factory import (
    build_model,
)

from train import (
    CFOTrainArgs,
    train_cfo,
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

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train bathymetry-conditioned CFO "
            "with an FNO2d backbone."
        )
    )

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
        default=32,
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
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default="FNO2d",
        choices=[
            "FNO2d",
        ],
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
        help=(
            "Seed used only for temporal "
            "snapshot selection."
        ),
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
            "bathy_cfo"
        ),
    )

    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="bathy_cfo",
    )

    parser.add_argument(
        "--running-ckpt-interval",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--running-ckpt-max-to-keep",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/"
            "bathy_cfo"
        ),
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
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    set_global_seed(
        args.seed
    )

    # ============================================================
    # DATASET PATH
    # ============================================================

    dataset_path = Path(
        args.dataset_path
    )

    if not dataset_path.is_absolute():

        dataset_path = (
            PROJECT_ROOT
            /
            dataset_path
        )

    # ============================================================
    # LOAD DATASET
    # ============================================================

    dataset = (
        load_bathymetry_dataset(
            dataset_path
        )
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

    # ============================================================
    # TIME
    # ============================================================

    original_train_snapshots = (
        train_data.shape[
            1
        ]
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
    # TEMPORAL SUBSAMPLING
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

    retained_train_snapshots = (
        train_data.shape[
            1
        ]
    )

    intervals_per_trajectory = (
        retained_train_snapshots
        -
        1
    )

    # ============================================================
    # SHAPES
    # ============================================================

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

    if (
        input_shape[
            -1
        ]
        !=
        3
    ):

        raise ValueError(
            "Expected q=[h,hu,hv] "
            "with 3 channels, "
            f"got {input_shape}."
        )

    if (
        condition_shape[
            -1
        ]
        !=
        1
    ):

        raise ValueError(
            "Expected one bathymetry channel, "
            f"got {condition_shape}."
        )

    # ============================================================
    # MODEL
    # ============================================================

    model = build_model(
        args.model,
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

    # ============================================================
    # SUMMARY
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "BATHYMETRY-CONDITIONED CFO"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: "
        f"{dataset_path}"
    )

    print(
        f"Model: "
        f"{args.model}"
    )

    print(
        f"State input shape: "
        f"{input_shape}"
    )

    print(
        f"Bathymetry shape: "
        f"{condition_shape}"
    )

    print(
        "Conditioning: "
        "N_theta(t, q, b)"
    )

    print(
        "Train/eval/test trajectories: "
        f"{len(train_data)}/"
        f"{len(eval_data)}/"
        f"{len(test_data)}"
    )

    print(
        "Original snapshots/trajectory: "
        f"{original_train_snapshots}"
    )

    print(
        "Retained snapshots/trajectory: "
        f"{retained_train_snapshots}"
    )

    print(
        "Spline intervals/trajectory: "
        f"{intervals_per_trajectory}"
    )

    print(
        "Temporal observation ratio: "
        f"{args.partial_train_ratio:.3f}"
    )

    print(
        "Model/training seed: "
        f"{args.seed}"
    )

    print(
        "Temporal-subsampling seed: "
        f"{args.partial_data_seed}"
    )

    print(
        f"Spline type: "
        f"{args.spline_type}"
    )

    print(
        f"gamma: "
        f"{args.gamma}"
    )

    print(
        "Physics loss: OFF"
    )

    print(
        "========================================\n"
    )

    if args.dry_run:

        return

    # ============================================================
    # CONDITIONED SPLINE DATASET
    # ============================================================

    print(
        "Preparing conditioned "
        "spline dataloader..."
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

    print(
        "Conditioned spline "
        "dataloader ready."
    )

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    running_ckpt_interval = (
        args.eval_interval
        if (
            args.running_ckpt_interval
            is None
        )
        else
        args.running_ckpt_interval
    )

    ckpt_root = Path(
        args.ckpt_dir
    )

    if not ckpt_root.is_absolute():

        ckpt_root = (
            PROJECT_ROOT
            /
            ckpt_root
        )

    # ============================================================
    # TRAINING ARGS
    # ============================================================

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
        learning_rate=float(
            args.lr
        ),
        beta1=float(
            args.beta1
        ),
        beta2=float(
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
            running_ckpt_interval
        ),

        running_ckpt_max_to_keep=(
            args.running_ckpt_max_to_keep
        ),

        best_ckpt_dir=str(
            ckpt_root
            /
            "best"
        ),

        best_ckpt_prefix=(
            args.ckpt_prefix
        ),
    )

    # ============================================================
    # VALIDATION
    # ============================================================

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

    print(
        "Starting Bathy-CFO training..."
    )

    train_output = train_cfo(
        method,
        spline_loader,
        train_args,
        eval_dataset=eval_dataset,
    )

    # ============================================================
    # SELECT STATE
    # ============================================================

    state_for_test = (
        train_output[
            "state"
        ]
    )

    if np.isfinite(
        train_output[
            "best_l2_error"
        ]
    ):

        state_for_test = (
            train_output[
                "best_state"
            ]
        )

    # ============================================================
    # ID TEST
    # ============================================================

    print(
        "\nRunning final ID "
        "test inference..."
    )

    test_pred = (
        method.uniform_inference(
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
    )

    # ============================================================
    # METRICS
    # ============================================================

    test_rel_l2 = (
        relative_L2_error(
            test_data,
            test_pred,
        )
    )

    test_rmse = rmse(
        test_data,
        test_pred,
    )

    test_rel_fro = (
        relative_frobenius_error(
            test_data,
            test_pred,
        )
    )

    print(
        "\n========================================"
    )

    print(
        "FINAL BATHY-CFO ID TEST METRICS"
    )

    print(
        "========================================"
    )

    print(
        "Relative L2 Error: "
        f"{float(test_rel_l2):.6f}"
    )

    print(
        "RMSE: "
        f"{float(test_rmse):.6f}"
    )

    print(
        "Relative Frobenius Error: "
        f"{float(test_rel_fro):.6f}"
    )

    if np.isfinite(
        train_output[
            "best_l2_error"
        ]
    ):

        print(
            "Best validation epoch: "
            f"{train_output['best_epoch']}"
        )

        print(
            "Best validation Rel L2: "
            f"{train_output['best_l2_error']:.6f}"
        )

    print(
        "========================================"
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_dir = Path(
        args.results_dir
    )

    if not results_dir.is_absolute():

        results_dir = (
            PROJECT_ROOT
            /
            results_dir
        )

    run_dir = (
        results_dir
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

        relative_l2=float(
            test_rel_l2
        ),

        rmse=float(
            test_rmse
        ),

        relative_frobenius=float(
            test_rel_fro
        ),

        best_eval_relative_l2=float(
            train_output[
                "best_l2_error"
            ]
        ),

        best_epoch=int(
            train_output[
                "best_epoch"
            ]
        ),

        seed=int(
            args.seed
        ),

        partial_train_ratio=float(
            args.partial_train_ratio
        ),

        partial_data_seed=int(
            args.partial_data_seed
        ),
    )

    print(
        f"Saved results to: "
        f"{run_dir}"
    )


if __name__ == "__main__":

    main()