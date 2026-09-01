"""CLI entrypoint for Physics-Informed CFO training experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


# ================================================================
# QUIET TENSORFLOW / ABSL LOGGING
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

from utils.readers import load_dataset_splits

from pi_cfo import PhysicsInformedCFO

from models.factory import build_model

from train import (
    CFOTrainArgs,
    train_cfo,
)

from utils.data import (
    build_dataloader,
    linear_spline,
    quintic_spline_batch,
    load_partial_data,
)

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)

from utils.seed import set_global_seed


# ================================================================
# COMMAND-LINE ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train Physics-Informed "
            "Continuous Flow Operator"
        )
    )

    # ------------------------------------------------------------
    # RUN SETUP
    # ------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=60_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Random seed for neural-network initialization, "
            "training-time sampling, noise, and dataloader shuffling."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    # ------------------------------------------------------------
    # OPTIMIZATION
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # CFO STOCHASTIC PATH
    # ------------------------------------------------------------

    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-5,
    )

    # ------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default="swe_full_small",
        choices=[
            "lorenz",
            "burgers",
            "dr",
            "swe",
            "swe_full_small",
        ],
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        default="UNet2D",
        choices=[
            "UNet1D",
            "UNet2D",
            "FNO1d",
            "FNO2d",
            "DiT",
            "SimpleMLP",
        ],
    )

    # ------------------------------------------------------------
    # SPLINE
    # ------------------------------------------------------------

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
        help=(
            "Batch size used only for spline "
            "coefficient construction."
        ),
    )

    # ============================================================
    # TEMPORAL SUBSAMPLING
    # ============================================================

    parser.add_argument(
        "--partial-train-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of training time snapshots retained "
            "before constructing temporal splines."
        ),
    )

    # ------------------------------------------------------------
    # IMPORTANT PHASE-40 ADDITION
    # ------------------------------------------------------------

    parser.add_argument(
        "--partial-data-seed",
        type=int,
        default=1234,
        help=(
            "Seed used ONLY to select temporal snapshots when "
            "--partial-train-ratio < 1. "
            "Keep fixed across model seeds during multi-seed "
            "experiments."
        ),
    )

    # ============================================================
    # PHYSICS-INFORMED PARAMETERS
    # ============================================================

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=1e-3,
        help=(
            "Weight multiplying the "
            "shallow-water PDE residual loss."
        ),
    )

    # ------------------------------------------------------------
    # Grid spacing for current 32 x 32 domain.
    #
    # Domain:
    #
    # [-2.5, 2.5]^2
    #
    # dx = dy = 5 / 32
    #          = 0.15625
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

    # ------------------------------------------------------------
    # EVALUATION / CHECKPOINTING
    # ------------------------------------------------------------

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="picfo_swe",
    )

    parser.add_argument(
        "--running-ckpt-interval",
        type=int,
        default=None,
        help=(
            "Save running checkpoint every N epochs. "
            "Defaults to eval-interval."
        ),
    )

    parser.add_argument(
        "--running-ckpt-max-to-keep",
        type=int,
        default=3,
        help=(
            "Number of running checkpoints to keep."
        ),
    )

    # ------------------------------------------------------------
    # LOGGING / UTILITY
    # ------------------------------------------------------------

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Disable periodic evaluation during training."
        ),
    )

    parser.add_argument(
        "--use-wandb",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print configuration and exit."
        ),
    )

    return parser.parse_args()


# ================================================================
# BUILD SPLINE DATASET
# ================================================================

def _build_spline_dataset(
    train_data,
    spline_type: str,
    batch_size: int,
    spline_batch_size: int,
    epochs: int,
    seed: int,
    time=None,
):

    batch_n, traj_len = (
        train_data.shape[:2]
    )

    # ------------------------------------------------------------
    # Full trajectory -> uniform normalized time.
    #
    # Partial trajectory -> use true selected timestamps.
    # ------------------------------------------------------------

    if time is None:

        time = np.broadcast_to(
            np.linspace(
                0.0,
                1.0,
                traj_len,
                dtype=train_data.dtype,
            ),
            (
                batch_n,
                traj_len,
            ),
        )

    # ------------------------------------------------------------
    # Construct temporal spline.
    # ------------------------------------------------------------

    if spline_type == "linear":

        (
            spline_coef,
            start_time,
            end_time,
        ) = linear_spline(
            train_data,
            time=time,
        )

    else:

        (
            spline_coef,
            start_time,
            end_time,
        ) = quintic_spline_batch(
            train_data,
            time=time,
            batch_size=spline_batch_size,
        )

    # ------------------------------------------------------------
    # Build training dataloader.
    # ------------------------------------------------------------

    return build_dataloader(
        spline_coef,
        t1=start_time,
        t2=end_time,
        batch_size=batch_size,
        num_epochs=epochs,
        seed=seed,
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    args = parse_args()

    # ============================================================
    # GLOBAL MODEL/TRAINING SEED
    # ============================================================

    set_global_seed(
        args.seed
    )

    # ------------------------------------------------------------
    # Current flat-bed PI-CFO experiment is unconditioned.
    # Bathymetry conditioning comes later.
    # ------------------------------------------------------------

    use_condition = False

    effective_lr = float(
        args.lr
    )

    effective_beta1 = float(
        args.beta1
    )

    effective_beta2 = float(
        args.beta2
    )

    # ============================================================
    # DRY RUN
    # ============================================================

    if args.dry_run:

        print(args)

        return

    # ============================================================
    # LOAD DATASET
    # ============================================================

    splits = load_dataset_splits(
        args.dataset,
        file_path=args.dataset_path,
    )

    train_data = splits[
        "train"
    ]

    eval_data = splits[
        "eval"
    ]

    test_data = splits[
        "test"
    ]

    # ============================================================
    # TEMPORAL SUBSAMPLING
    # ============================================================

    train_time = None

    original_train_snapshots = (
        train_data.shape[1]
    )

    if args.partial_train_ratio < 1.0:

        (
            train_data,
            train_time,
        ) = load_partial_data(
            train_data,
            ratio=args.partial_train_ratio,

            # ----------------------------------------------------
            # IMPORTANT PHASE-40 CHANGE
            # ----------------------------------------------------
            seed=args.partial_data_seed,
        )

    retained_train_snapshots = (
        train_data.shape[1]
    )

    # ============================================================
    # INPUT SHAPE
    # ============================================================

    input_shape = tuple(
        train_data.shape[2:]
    )

    # Expected:
    #
    # (32, 32, 3)
    #
    # q = [h, hu, hv]

    # ============================================================
    # BUILD MODEL
    # ============================================================

    model = build_model(
        args.model,
        input_shape,
        use_condition=use_condition,
    )

    # ============================================================
    # TASK SUMMARY
    # ============================================================

    task_desc = (
        "partial snapshots"
        if args.partial_train_ratio < 1.0
        else "full trajectories"
    )

    print(
        "\n========================================"
    )

    print(
        "PHYSICS-INFORMED CFO TASK SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Model: {args.model}"
    )

    print(
        f"Spline type: {args.spline_type}"
    )

    print(
        f"Observation type: {task_desc}"
    )

    print(
        f"Temporal observation ratio: "
        f"{args.partial_train_ratio:.3f}"
    )

    print(
        f"Original snapshots/trajectory: "
        f"{original_train_snapshots}"
    )

    print(
        f"Retained snapshots/trajectory: "
        f"{retained_train_snapshots}"
    )

    print(
        f"Model/training seed: "
        f"{args.seed}"
    )

    print(
        f"Temporal-subsampling seed: "
        f"{args.partial_data_seed}"
    )

    print(
        f"Input shape: {input_shape}"
    )

    print(
        f"lambda_PDE: "
        f"{args.lambda_pde}"
    )

    print(
        f"dx: {args.dx}"
    )

    print(
        f"dy: {args.dy}"
    )

    print(
        f"gravity: {args.gravity}"
    )

    print(
        "========================================\n"
    )

    # ============================================================
    # TRAINING SETTINGS
    # ============================================================

    print(
        "=== PI-CFO Training Run ==="
    )

    print(
        f"seed={args.seed}"
    )

    print(
        f"epochs={args.epochs}"
    )

    print(
        f"batch_size={args.batch_size}"
    )

    print(
        f"spline_batch_size="
        f"{args.spline_batch_size}"
    )

    print(
        f"lr={effective_lr}"
    )

    print(
        f"beta1={effective_beta1}"
    )

    print(
        f"beta2={effective_beta2}"
    )

    print(
        f"gamma={args.gamma}"
    )

    print(
        f"lambda_pde="
        f"{args.lambda_pde}"
    )

    print(
        f"eval_interval="
        f"{args.eval_interval}"
    )

    print(
        f"do_eval="
        f"{not args.no_eval}"
    )

    print(
        "train/eval/test sizes = "
        f"{len(train_data)}/"
        f"{len(eval_data)}/"
        f"{len(test_data)}"
    )

    print(
        f"input_shape={input_shape}"
    )

    # ============================================================
    # CHECKPOINT POLICY
    # ============================================================

    running_ckpt_interval = (
        args.eval_interval
        if args.running_ckpt_interval is None
        else args.running_ckpt_interval
    )

    print(
        "checkpoint_policy "
        f"running_every="
        f"{running_ckpt_interval} "
        f"keep="
        f"{args.running_ckpt_max_to_keep} "
        "best_keep=1"
    )

    # ============================================================
    # BUILD TEMPORAL SPLINES
    # ============================================================

    print(
        "\nPreparing spline dataloader..."
    )

    spline_loader = _build_spline_dataset(
        train_data=train_data,
        spline_type=args.spline_type,
        batch_size=args.batch_size,
        spline_batch_size=args.spline_batch_size,
        epochs=args.epochs,

        # --------------------------------------------------------
        # Model/training seed controls dataloader shuffle.
        # --------------------------------------------------------
        seed=args.seed,

        # --------------------------------------------------------
        # If partial data were selected, these are the actual
        # normalized original timestamps, not a re-uniformized
        # 0...1 grid over the reduced sequence.
        # --------------------------------------------------------
        time=train_time,
    )

    print(
        "Spline construction complete."
    )

    # ============================================================
    # CREATE PI-CFO METHOD
    # ============================================================

    method = PhysicsInformedCFO(
        model=model,
        input_shape=input_shape,
        gamma=float(
            args.gamma
        ),
        spline_type=args.spline_type,
        use_condition=use_condition,

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

    # ============================================================
    # TRAINING ARGUMENTS
    # ============================================================

    train_args = CFOTrainArgs(
        num_epochs=args.epochs,
        random_seed=args.seed,
        use_wandb=args.use_wandb,

        learning_rate=effective_lr,

        beta1=effective_beta1,

        beta2=effective_beta2,

        do_eval=not args.no_eval,

        eval_interval=args.eval_interval,

        irregular_time=False,

        running_ckpt_dir=str(
            (
                Path(
                    args.ckpt_dir
                ).resolve()
                /
                "running"
            )
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
            (
                Path(
                    args.ckpt_dir
                ).resolve()
                /
                "best"
            )
        ),

        best_ckpt_prefix=(
            args.ckpt_prefix
        ),
    )

    # ============================================================
    # VALIDATION
    #
    # Even sparse-data models are evaluated against the full
    # validation trajectories.
    # ============================================================

    eval_dataset = (
        eval_data[:, 0],
        eval_data,
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print(
        "\nStarting PI-CFO training loop..."
    )

    train_output = train_cfo(
        method,
        spline_loader,
        train_args,
        eval_dataset=eval_dataset,
    )

    # ============================================================
    # SELECT BEST MODEL
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
    # FINAL TEST INFERENCE
    #
    # IMPORTANT:
    # Always evaluate on all 51 test snapshots, even when the
    # model was trained with only ~13 temporal snapshots.
    # ============================================================

    print(
        "\nRunning final PI-CFO "
        "test inference..."
    )

    test_pred = (
        method.uniform_inference(
            state_for_test,
            test_data[:, 0],
            trajectory_points_num=(
                test_data.shape[1]
            ),
            steps_per_segment=2,
            method="RK4",
        )
    )

    # ============================================================
    # FINAL TEST METRICS
    # ============================================================

    test_rmse = rmse(
        test_data,
        test_pred,
    )

    test_rel_l2 = (
        relative_L2_error(
            test_data,
            test_pred,
        )
    )

    test_rel_fro = (
        relative_frobenius_error(
            test_data,
            test_pred,
        )
    )

    # ============================================================
    # BEST VALIDATION CHECKPOINT
    # ============================================================

    best_epoch = (
        train_output[
            "best_epoch"
        ]
    )

    best_rel_l2 = (
        train_output[
            "best_l2_error"
        ]
    )

    if np.isfinite(
        best_rel_l2
    ):

        print(
            "\nBest eval checkpoint:"
        )

        print(
            f"  epoch="
            f"{best_epoch}"
        )

        print(
            f"  rel_l2="
            f"{best_rel_l2:.6f}"
        )

    else:

        print(
            "\nBest eval checkpoint: "
            "none "
            "(evaluation disabled "
            "or not run)"
        )

    # ============================================================
    # FINAL PI-CFO RESULTS
    # ============================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL PI-CFO TEST METRICS"
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

    # ------------------------------------------------------------
    # Phase-29 loss-component information, if present.
    # ------------------------------------------------------------

    if (
        "total_loss_log"
        in train_output
        and
        len(
            train_output[
                "total_loss_log"
            ]
        )
        > 0
    ):

        print(
            "\nFinal training losses:"
        )

        print(
            "  Total: "
            f"{train_output['total_loss_log'][-1]:.6e}"
        )

        print(
            "  CFO:   "
            f"{train_output['cfo_loss_log'][-1]:.6e}"
        )

        if (
            len(
                train_output[
                    "pde_loss_log"
                ]
            )
            > 0
        ):

            print(
                "  PDE:   "
                f"{train_output['pde_loss_log'][-1]:.6e}"
            )

        if (
            "weighted_pde_loss_log"
            in train_output
            and
            len(
                train_output[
                    "weighted_pde_loss_log"
                ]
            )
            > 0
        ):

            print(
                "  Weighted PDE: "
                f"{train_output['weighted_pde_loss_log'][-1]:.6e}"
            )

    print(
        "========================================"
    )

    # ============================================================
    # CHECKPOINT LOCATIONS
    # ============================================================

    ckpt_dir = (
        Path(
            args.ckpt_dir
        ).resolve()
    )

    print(
        "\nRunning checkpoints:"
    )

    print(
        f"  dir="
        f"{ckpt_dir / 'running'}"
    )

    print(
        f"  keep="
        f"{args.running_ckpt_max_to_keep}"
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        f"  dir="
        f"{ckpt_dir / 'best'}"
    )

    print(
        "  keep=1"
    )


# ================================================================
# SCRIPT ENTRYPOINT
# ================================================================

if __name__ == "__main__":
    main()