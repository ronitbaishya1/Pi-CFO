"""
Train-and-Unroll training for SC-FNO-v3 Bed-PI-CFO.

Training sequence
-----------------
Stage 1:
    K = 1
    train from random initialization

Stage 2:
    K = 2
    initialize MODEL PARAMETERS from best K=1 checkpoint

Stage 3:
    K = 3
    initialize MODEL PARAMETERS from best K=2 checkpoint

Stage 4:
    K = 4
    initialize MODEL PARAMETERS from best K=3 checkpoint

Important
---------
The optimizer is RESET at every new composition depth.

Only the learned neural-network parameters are transferred.

Physics objective
-----------------

    L_total
    =
    L_CFO
    +
    lambda_PDE * L_PDE
    +
    lambda_bed * L_bed

Default values:

    lambda_PDE = 0.03
    lambda_bed = 0.70

No well-balanced loss is used yet.

Architecture
------------

    SCFNO2dV3

with residual self-composition:

    z_{k+1}
    =
    z_k
    +
    alpha * Phi_theta(z_k)

Default:

    alpha = 0.25

Recommended first experiment
----------------------------

    K=1 : 1000 epochs
    K=2 :  500 epochs
    K=3 :  500 epochs
    K=4 :  500 epochs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from jax import value_and_grad
from tqdm.auto import trange


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

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from models.sc_fno_v3 import (
    SCFNO2dV3,
)

from train import (
    init_cfo_train_state,
    _prepare_cfo_train_batch,
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
    prepare_tf_data,
    prefetch_to_device,
)

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)

from utils.checkpoints import (
    load_train_state,
    save_train_state,
)

from utils.seed import (
    set_global_seed,
)


# ================================================================
# COMMAND-LINE ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train-and-Unroll stage for "
            "SC-FNO-v3 Bed-PI-CFO."
        )
    )

    # ============================================================
    # STAGE
    # ============================================================

    parser.add_argument(
        "--compose-depth",
        type=int,
        required=True,
        choices=[
            1,
            2,
            3,
            4,
        ],
        help=(
            "Current self-composition depth K."
        ),
    )

    parser.add_argument(
        "--warm-start-dir",
        type=str,
        default=None,
        help=(
            "Best checkpoint directory from the "
            "previous composition depth. "
            "Leave empty for K=1."
        ),
    )

    parser.add_argument(
        "--warm-start-prefix",
        type=str,
        default="sc_v3_bathy_bed_pi",
        help=(
            "Checkpoint prefix used in the "
            "previous stage."
        ),
    )

    # ============================================================
    # TRAINING
    # ============================================================

    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
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

    # ============================================================
    # CFO
    # ============================================================

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

    # ============================================================
    # PHYSICS
    # ============================================================

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--lambda-bed",
        type=float,
        default=0.70,
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

    # ============================================================
    # SC-FNO-v3
    # ============================================================

    parser.add_argument(
        "--modes1",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--modes2",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
    )

    # ============================================================
    # DATA
    # ============================================================

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
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

    # ============================================================
    # EVALUATION
    # ============================================================

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    # ============================================================
    # CHECKPOINTS
    # ============================================================

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "sc_v3_bathy_bed_pi"
        ),
    )

    parser.add_argument(
        "--ckpt-prefix",
        type=str,
        default="sc_v3_bathy_bed_pi",
    )

    parser.add_argument(
        "--running-max-to-keep",
        type=int,
        default=3,
    )

    # ============================================================
    # RESULTS
    # ============================================================

    parser.add_argument(
        "--results-dir",
        type=str,
        default=(
            "results/"
            "sc_v3_bathy_bed_pi"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser.parse_args()


# ================================================================
# PATH
# ================================================================

def resolve_path(
    value,
):

    if value is None:

        return None

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
# PARAMETER COUNT
# ================================================================

def count_parameters(
    params,
):

    return int(
        sum(
            leaf.size
            for leaf
            in jax.tree_util.tree_leaves(
                params
            )
        )
    )


# ================================================================
# PARAMETER-TREE COMPATIBILITY
# ================================================================

def assert_parameter_compatibility(
    target_params,
    source_params,
):

    target_structure = (
        jax.tree_util.tree_structure(
            target_params
        )
    )

    source_structure = (
        jax.tree_util.tree_structure(
            source_params
        )
    )

    if (
        target_structure
        !=
        source_structure
    ):

        raise RuntimeError(
            "Warm-start parameter trees "
            "have different structures."
        )

    target_leaves = (
        jax.tree_util.tree_leaves(
            target_params
        )
    )

    source_leaves = (
        jax.tree_util.tree_leaves(
            source_params
        )
    )

    if (
        len(
            target_leaves
        )
        !=
        len(
            source_leaves
        )
    ):

        raise RuntimeError(
            "Warm-start parameter trees "
            "contain different numbers "
            "of leaves."
        )

    for index, (
        target_leaf,
        source_leaf,
    ) in enumerate(
        zip(
            target_leaves,
            source_leaves,
        )
    ):

        if (
            target_leaf.shape
            !=
            source_leaf.shape
        ):

            raise RuntimeError(
                "Warm-start shape mismatch "
                f"at parameter leaf {index}: "
                f"target={target_leaf.shape}, "
                f"source={source_leaf.shape}."
            )


# ================================================================
# LOAD WARM-START PARAMETERS
# ================================================================

def warm_start_parameters(
    *,
    target_state,
    checkpoint_dir,
    checkpoint_prefix,
):

    checkpoint_dir = Path(
        checkpoint_dir
    )

    if not checkpoint_dir.exists():

        raise FileNotFoundError(
            "Warm-start checkpoint directory "
            f"does not exist:\n"
            f"{checkpoint_dir}"
        )

    print(
        "\nLoading previous-stage checkpoint..."
    )

    restored_state = load_train_state(
        target_state,
        str(
            checkpoint_dir
        ),
        prefix=(
            checkpoint_prefix
        ),
        step=None,
        max_to_keep=1,
    )

    assert_parameter_compatibility(
        target_state.params,
        restored_state.params,
    )

    # ============================================================
    # IMPORTANT:
    #
    # We transfer ONLY neural-network parameters.
    #
    # target_state contains a freshly initialized Adam optimizer,
    # so the optimizer moments are RESET for the new depth.
    # ============================================================

    new_state = target_state.replace(
        params=(
            restored_state.params
        )
    )

    print(
        "Warm start successful."
    )

    print(
        "Transferred: model parameters"
    )

    print(
        "Reset:       Adam optimizer state"
    )

    return new_state


# ================================================================
# LOSS-COMPONENT PARSER
# ================================================================

def unpack_bed_loss_components(
    components,
):

    """
    Expected Bed-PI-CFO outputs:

        total_loss
        cfo_loss
        pde_loss
        bed_loss

    Some local versions may return extra diagnostic quantities.
    We deliberately use the first four entries.
    """

    if not isinstance(
        components,
        (
            tuple,
            list,
        ),
    ):

        raise TypeError(
            "method.loss_components() must "
            "return a tuple/list."
        )

    if len(
        components
    ) < 4:

        raise RuntimeError(
            "Bed-PI-CFO loss_components() "
            "must return at least four values:\n"
            "total, CFO, PDE, bed."
        )

    total_loss = components[
        0
    ]

    cfo_loss = components[
        1
    ]

    pde_loss = components[
        2
    ]

    bed_loss = components[
        3
    ]

    return (
        total_loss,
        cfo_loss,
        pde_loss,
        bed_loss,
    )


# ================================================================
# EVALUATION
# ================================================================

def evaluate_model(
    method,
    state,
    q_true,
    bathymetry,
    *,
    steps_per_segment,
):

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
        steps_per_segment=(
            steps_per_segment
        ),
        condition=(
            bathymetry
        ),
        method="RK4",
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    rel_l2 = float(
        relative_L2_error(
            q_true,
            prediction,
        )
    )

    rmse_value = float(
        rmse(
            q_true,
            prediction,
        )
    )

    rel_fro = float(
        relative_frobenius_error(
            q_true,
            prediction,
        )
    )

    return {
        "prediction":
            prediction,

        "relative_l2":
            rel_l2,

        "rmse":
            rmse_value,

        "relative_frobenius":
            rel_fro,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    # ============================================================
    # BASIC VALIDATION
    # ============================================================

    if (
        args.compose_depth
        ==
        1
        and
        args.warm_start_dir
        is not None
    ):

        raise ValueError(
            "K=1 must train from scratch. "
            "Do not supply --warm-start-dir."
        )

    if (
        args.compose_depth
        >
        1
        and
        args.warm_start_dir
        is None
    ):

        raise ValueError(
            "K>1 requires --warm-start-dir "
            "from the previous depth."
        )

    if args.alpha <= 0.0:

        raise ValueError(
            "--alpha must be positive."
        )

    if args.epochs < 1:

        raise ValueError(
            "--epochs must be >= 1."
        )

    if args.eval_interval < 1:

        raise ValueError(
            "--eval-interval must be >= 1."
        )

    # ============================================================
    # SEED
    # ============================================================

    set_global_seed(
        args.seed
    )

    # ============================================================
    # PATHS
    # ============================================================

    dataset_path = resolve_path(
        args.dataset_path
    )

    checkpoint_root = (
        resolve_path(
            args.ckpt_dir
        )
        /
        f"depth{args.compose_depth}"
        /
        f"seed{args.seed}"
    )

    best_checkpoint_dir = (
        checkpoint_root
        /
        "best"
    )

    running_checkpoint_dir = (
        checkpoint_root
        /
        "running"
    )

    results_root = (
        resolve_path(
            args.results_dir
        )
        /
        f"depth{args.compose_depth}"
        /
        f"seed{args.seed}"
    )

    warm_start_dir = resolve_path(
        args.warm_start_dir
    )

    # ============================================================
    # LOAD DATA
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

    train_bathymetry = np.asarray(
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

    eval_bathymetry = np.asarray(
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

    test_bathymetry = np.asarray(
        test_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    # ============================================================
    # TRAIN TIME
    # ============================================================

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
            dtype=(
                train_data.dtype
            ),
        )
    )

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

    # ============================================================
    # SHAPES
    # ============================================================

    input_shape = tuple(
        train_data.shape[
            2:
        ]
    )

    condition_shape = tuple(
        train_bathymetry.shape[
            1:
        ]
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = SCFNO2dV3(
        num_channels=(
            input_shape[
                -1
            ]
        ),
        modes1=(
            args.modes1
        ),
        modes2=(
            args.modes2
        ),
        width=(
            args.width
        ),
        compose_depth=(
            args.compose_depth
        ),
        alpha=(
            args.alpha
        ),
        use_condition=True,
        use_time=True,
    )

    # ============================================================
    # BED-PI-CFO METHOD
    # ============================================================

    method = BathymetryBedRegularizedPICFO(
        model=model,
        input_shape=input_shape,
        condition_shape=(
            condition_shape
        ),
        gamma=(
            args.gamma
        ),
        spline_type=(
            args.spline_type
        ),
        lambda_pde=(
            args.lambda_pde
        ),
        lambda_bed=(
            args.lambda_bed
        ),
        dx=(
            args.dx
        ),
        dy=(
            args.dy
        ),
        gravity=(
            args.gravity
        ),
    )

    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================

    print(
        "\n"
        +
        "=" * 55
    )

    print(
        "SC-FNO-v3 BED-PI-CFO — TRAIN AND UNROLL"
    )

    print(
        "=" * 55
    )

    print(
        f"Current stage K:        "
        f"{args.compose_depth}"
    )

    print(
        f"Epochs this stage:      "
        f"{args.epochs}"
    )

    print(
        f"alpha:                  "
        f"{args.alpha}"
    )

    print(
        f"lambda_PDE:             "
        f"{args.lambda_pde}"
    )

    print(
        f"lambda_bed:             "
        f"{args.lambda_bed}"
    )

    print(
        f"Learning rate:          "
        f"{args.lr}"
    )

    print(
        f"Train/eval/test:        "
        f"{train_data.shape[0]}/"
        f"{eval_data.shape[0]}/"
        f"{test_data.shape[0]}"
    )

    print(
        f"State shape:            "
        f"{input_shape}"
    )

    print(
        f"Bathymetry shape:       "
        f"{condition_shape}"
    )

    if warm_start_dir is None:

        print(
            "Initialization:          RANDOM"
        )

    else:

        print(
            "Initialization:          "
            "PREVIOUS STAGE"
        )

        print(
            f"Warm-start directory:   "
            f"{warm_start_dir}"
        )

    print(
        "\nObjective:"
    )

    print(
        "L = L_CFO "
        "+ 0.03 L_PDE "
        "+ 0.70 L_bed"
    )

    print(
        "\nWell-balanced loss: OFF"
    )

    print(
        "=" * 55
    )

    if args.dry_run:

        return

    # ============================================================
    # SPLINE DATALOADER
    # ============================================================

    print(
        "\nPreparing conditioned "
        "spline dataloader..."
    )

    spline_loader = (
        build_conditioned_spline_dataloader(
            train_data=(
                train_data
            ),
            bathymetry=(
                train_bathymetry
            ),
            time=(
                train_time
            ),
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
    # INITIAL TRAIN STATE
    # ============================================================

    target_state = init_cfo_train_state(
        method,
        seed=(
            args.seed
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
    )

    # ============================================================
    # RANDOM K=1 OR WARM-START K>1
    # ============================================================

    if warm_start_dir is None:

        state = target_state

        print(
            "\nStarting from random "
            "initialization."
        )

    else:

        state = warm_start_parameters(
            target_state=(
                target_state
            ),
            checkpoint_dir=(
                warm_start_dir
            ),
            checkpoint_prefix=(
                args.warm_start_prefix
            ),
        )

    print(
        f"Model parameters: "
        f"{count_parameters(state.params):,}"
    )

    # ============================================================
    # TRAINING LOSS
    # ============================================================

    def component_loss_fn(
        params,
        batch,
    ):

        components = (
            method.loss_components(
                params,
                batch,
            )
        )

        (
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
        ) = unpack_bed_loss_components(
            components
        )

        return (
            total_loss,
            (
                cfo_loss,
                pde_loss,
                bed_loss,
            ),
        )

    # ============================================================
    # ONE OPTIMIZATION STEP
    # ============================================================

    def train_step(
        current_state,
        batch,
    ):

        (
            (
                total_loss,
                (
                    cfo_loss,
                    pde_loss,
                    bed_loss,
                ),
            ),
            gradients,
        ) = value_and_grad(
            component_loss_fn,
            has_aux=True,
        )(
            current_state.params,
            batch,
        )

        next_state = (
            current_state.apply_gradients(
                grads=(
                    gradients
                )
            )
        )

        return (
            next_state,
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
        )

    train_step_jit = jax.jit(
        train_step
    )

    # ============================================================
    # DATA PIPELINE
    # ============================================================

    data_iterator = map(
        prepare_tf_data,
        spline_loader,
    )

    data_iterator = prefetch_to_device(
        data_iterator,
        2,
    )

    # ============================================================
    # RANDOM KEYS
    # ============================================================

    rng_key = jax.random.PRNGKey(
        args.seed
    )

    # ============================================================
    # LOGS
    # ============================================================

    total_loss_log = []

    cfo_loss_log = []

    pde_loss_log = []

    bed_loss_log = []

    weighted_pde_log = []

    weighted_bed_log = []

    eval_epoch_log = []

    eval_l2_log = []

    eval_rmse_log = []

    eval_fro_log = []

    # ============================================================
    # BEST MODEL
    # ============================================================

    best_state = state

    best_l2 = float(
        "inf"
    )

    best_epoch = -1

    # ============================================================
    # TRAIN
    # ============================================================

    print(
        "\nStarting training..."
    )

    progress = trange(
        args.epochs,
        desc=(
            f"SC-v3 K={args.compose_depth}"
        ),
    )

    for epoch_index in progress:

        # --------------------------------------------------------
        # Random time/noise keys
        # --------------------------------------------------------

        (
            rng_key,
            time_key,
            noise_key,
        ) = jax.random.split(
            rng_key,
            3,
        )

        # --------------------------------------------------------
        # CFO random-time training batch
        # --------------------------------------------------------

        batch = _prepare_cfo_train_batch(
            method,
            next(
                data_iterator
            ),
            time_key,
            noise_key,
        )

        # --------------------------------------------------------
        # Gradient update
        # --------------------------------------------------------

        (
            state,
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
        ) = train_step_jit(
            state,
            batch,
        )

        total_value = float(
            total_loss
        )

        cfo_value = float(
            cfo_loss
        )

        pde_value = float(
            pde_loss
        )

        bed_value = float(
            bed_loss
        )

        weighted_pde_value = (
            float(
                args.lambda_pde
            )
            *
            pde_value
        )

        weighted_bed_value = (
            float(
                args.lambda_bed
            )
            *
            bed_value
        )

        total_loss_log.append(
            total_value
        )

        cfo_loss_log.append(
            cfo_value
        )

        pde_loss_log.append(
            pde_value
        )

        bed_loss_log.append(
            bed_value
        )

        weighted_pde_log.append(
            weighted_pde_value
        )

        weighted_bed_log.append(
            weighted_bed_value
        )

        progress.set_postfix(
            {
                "total":
                    f"{total_value:.3e}",

                "cfo":
                    f"{cfo_value:.3e}",

                "pde":
                    f"{pde_value:.3e}",

                "bed":
                    f"{bed_value:.3e}",
            }
        )

        # ========================================================
        # EVALUATE
        # ========================================================

        stage_epoch = (
            epoch_index
            +
            1
        )

        should_evaluate = (
            stage_epoch
            %
            args.eval_interval
            ==
            0
            or
            stage_epoch
            ==
            args.epochs
        )

        if should_evaluate:

            print(
                "\n"
                f"[K={args.compose_depth}] "
                f"evaluation at stage epoch "
                f"{stage_epoch}"
            )

            evaluation = evaluate_model(
                method,
                state,
                eval_data,
                eval_bathymetry,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

            current_l2 = evaluation[
                "relative_l2"
            ]

            current_rmse = evaluation[
                "rmse"
            ]

            current_fro = evaluation[
                "relative_frobenius"
            ]

            eval_epoch_log.append(
                stage_epoch
            )

            eval_l2_log.append(
                current_l2
            )

            eval_rmse_log.append(
                current_rmse
            )

            eval_fro_log.append(
                current_fro
            )

            print(
                f"Validation Rel L2: "
                f"{current_l2:.6f}"
            )

            print(
                f"Validation RMSE:   "
                f"{current_rmse:.6f}"
            )

            print(
                f"Validation Rel Fro:"
                f" {current_fro:.6f}"
            )

            # ----------------------------------------------------
            # Running checkpoint
            # ----------------------------------------------------

            save_train_state(
                state,
                str(
                    running_checkpoint_dir
                ),
                prefix=(
                    args.ckpt_prefix
                ),
                step=(
                    stage_epoch
                ),
                max_to_keep=(
                    args.running_max_to_keep
                ),
            )

            # ----------------------------------------------------
            # Best checkpoint
            # ----------------------------------------------------

            if (
                current_l2
                <
                best_l2
            ):

                best_l2 = (
                    current_l2
                )

                best_epoch = (
                    stage_epoch
                )

                best_state = (
                    state
                )

                save_train_state(
                    best_state,
                    str(
                        best_checkpoint_dir
                    ),
                    prefix=(
                        args.ckpt_prefix
                    ),
                    step=(
                        stage_epoch
                    ),
                    max_to_keep=1,
                )

                print(
                    "NEW BEST CHECKPOINT"
                )

                print(
                    f"Best Rel L2 = "
                    f"{best_l2:.6f}"
                )

    # ============================================================
    # FINAL TRAINING LOSSES
    # ============================================================

    print(
        "\n"
        +
        "=" * 55
    )

    print(
        "FINAL STAGE TRAINING LOSSES"
    )

    print(
        "=" * 55
    )

    print(
        f"Total:        "
        f"{total_loss_log[-1]:.6e}"
    )

    print(
        f"CFO:          "
        f"{cfo_loss_log[-1]:.6e}"
    )

    print(
        f"PDE:          "
        f"{pde_loss_log[-1]:.6e}"
    )

    print(
        f"Weighted PDE: "
        f"{weighted_pde_log[-1]:.6e}"
    )

    print(
        f"Bed:          "
        f"{bed_loss_log[-1]:.6e}"
    )

    print(
        f"Weighted Bed: "
        f"{weighted_bed_log[-1]:.6e}"
    )

    # ============================================================
    # ID TEST USING BEST CHECKPOINT IN MEMORY
    # ============================================================

    print(
        "\nRunning ID test with "
        "best stage checkpoint..."
    )

    test_evaluation = evaluate_model(
        method,
        best_state,
        test_data,
        test_bathymetry,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    test_prediction = test_evaluation[
        "prediction"
    ]

    test_l2 = test_evaluation[
        "relative_l2"
    ]

    test_rmse_value = test_evaluation[
        "rmse"
    ]

    test_fro = test_evaluation[
        "relative_frobenius"
    ]

    # ============================================================
    # PRINT FINAL RESULTS
    # ============================================================

    print(
        "\n"
        +
        "=" * 55
    )

    print(
        "SC-FNO-v3 BED-PI-CFO ID TEST"
    )

    print(
        "=" * 55
    )

    print(
        f"Composition depth K: "
        f"{args.compose_depth}"
    )

    print(
        f"Relative L2:         "
        f"{test_l2:.6f}"
    )

    print(
        f"RMSE:                "
        f"{test_rmse_value:.6f}"
    )

    print(
        f"Rel Fro:             "
        f"{test_fro:.6f}"
    )

    print(
        f"Best stage epoch:    "
        f"{best_epoch}"
    )

    print(
        f"Best validation L2:  "
        f"{best_l2:.6f}"
    )

    print(
        "=" * 55
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        results_root
        /
        "test_true.npy",
        test_data,
    )

    np.save(
        results_root
        /
        "test_pred.npy",
        test_prediction,
    )

    np.save(
        results_root
        /
        "test_bathymetry.npy",
        test_bathymetry,
    )

    np.savez(
        results_root
        /
        "training_history.npz",

        total_loss=np.asarray(
            total_loss_log
        ),

        cfo_loss=np.asarray(
            cfo_loss_log
        ),

        pde_loss=np.asarray(
            pde_loss_log
        ),

        bed_loss=np.asarray(
            bed_loss_log
        ),

        weighted_pde_loss=np.asarray(
            weighted_pde_log
        ),

        weighted_bed_loss=np.asarray(
            weighted_bed_log
        ),

        eval_epoch=np.asarray(
            eval_epoch_log
        ),

        eval_relative_l2=np.asarray(
            eval_l2_log
        ),

        eval_rmse=np.asarray(
            eval_rmse_log
        ),

        eval_relative_frobenius=np.asarray(
            eval_fro_log
        ),
    )

    np.savez(
        results_root
        /
        "metrics.npz",

        compose_depth=int(
            args.compose_depth
        ),

        alpha=float(
            args.alpha
        ),

        epochs=int(
            args.epochs
        ),

        lambda_pde=float(
            args.lambda_pde
        ),

        lambda_bed=float(
            args.lambda_bed
        ),

        best_epoch=int(
            best_epoch
        ),

        best_eval_relative_l2=float(
            best_l2
        ),

        test_relative_l2=float(
            test_l2
        ),

        test_rmse=float(
            test_rmse_value
        ),

        test_relative_frobenius=float(
            test_fro
        ),

        final_total_loss=float(
            total_loss_log[
                -1
            ]
        ),

        final_cfo_loss=float(
            cfo_loss_log[
                -1
            ]
        ),

        final_pde_loss=float(
            pde_loss_log[
                -1
            ]
        ),

        final_bed_loss=float(
            bed_loss_log[
                -1
            ]
        ),
    )

    print(
        f"\nResults saved to:\n"
        f"{results_root}"
    )

    print(
        f"\nBest checkpoint saved to:\n"
        f"{best_checkpoint_dir}"
    )


if __name__ == "__main__":

    main()