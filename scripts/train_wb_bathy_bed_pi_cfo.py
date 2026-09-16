"""
Train Well-Balanced Bed-PI-CFO on the bathymetry-conditioned
2D shallow-water dataset.

Objective
---------
L_total
=
L_CFO
+
lambda_PDE * L_PDE
+
lambda_bed * L_bed
+
lambda_WB * L_WB

This script supports two backbones:

    fno
    local_global

For Phase A use:
    --architecture fno

For Phase B use:
    --architecture local_global

The original FNO files are not modified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax.tree_util import tree_leaves
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

from models.fno import FNO2d

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.data import (
    build_dataloader,
    load_partial_data,
    prepare_tf_data,
    prefetch_to_device,
)

from utils.splines import (
    linear_spline,
    quintic_spline_batch,
)

from utils.metrics import (
    relative_L2_error,
    relative_frobenius_error,
    rmse,
)

from utils.checkpoints import (
    save_train_state,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train FNO or Local-Global FNO "
            "Well-Balanced Bed-PI-CFO."
        )
    )

    # ------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_32_id.h5"
        ),
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

    # ------------------------------------------------------------
    # ARCHITECTURE
    # ------------------------------------------------------------

    parser.add_argument(
        "--architecture",
        type=str,
        default="fno",
        choices=[
            "fno",
            "local_global",
        ],
    )

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
        "--num-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--local-kernel-size",
        type=int,
        default=3,
    )

    # ------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--spline-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
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
        "--spline-type",
        type=str,
        default="quintic",
        choices=[
            "linear",
            "quintic",
        ],
    )

    # ------------------------------------------------------------
    # PHYSICS LOSSES
    # ------------------------------------------------------------

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
        "--lambda-wb",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
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
    # EVALUATION
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------

    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


# ================================================================
# FORMAT LAMBDA FOR DIRECTORY NAME
# ================================================================

def lambda_name(value):

    text = f"{float(value):g}"

    return (
        text
        .replace(
            ".",
            "p",
        )
        .replace(
            "-",
            "m",
        )
    )


# ================================================================
# NORMALIZE TIME ARRAY
# ================================================================

def trajectory_time_array(
    split,
):

    q = split[
        "q"
    ]

    time = np.asarray(
        split[
            "time"
        ],
        dtype=np.float32,
    )

    B = q.shape[0]
    T = q.shape[1]

    if time.ndim == 1:

        if time.shape[0] != T:

            raise ValueError(
                "1D time array length does not "
                "match trajectory length."
            )

        time = np.broadcast_to(
            time[
                None,
                :
            ],
            (
                B,
                T,
            ),
        ).copy()

    elif time.ndim == 2:

        if time.shape != (
            B,
            T,
        ):

            raise ValueError(
                f"Expected time shape {(B, T)}, "
                f"got {time.shape}."
            )

    else:

        raise ValueError(
            "time must have shape (T,) "
            "or (B,T)."
        )

    # CFO inference assumes normalized [0,1].
    t0 = time[
        :,
        0:1
    ]

    t1 = time[
        :,
        -1:
    ]

    duration = np.maximum(
        t1 - t0,
        1e-12,
    )

    return (
        time
        -
        t0
    ) / duration


# ================================================================
# BUILD MODEL
# ================================================================

def build_backbone(
    args,
    num_channels,
):

    if args.architecture == "fno":

        return FNO2d(
            num_channels=num_channels,
            modes1=args.modes1,
            modes2=args.modes2,
            width=args.width,
            num_blocks=args.num_blocks,
            use_condition=True,
            use_time=True,
        )

    # Import only when requested so Phase A works
    # before local_global_fno.py exists.
    from models.local_global_fno import (
        LocalGlobalFNO2d,
    )

    return LocalGlobalFNO2d(
        num_channels=num_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_blocks=args.num_blocks,
        local_kernel_size=(
            args.local_kernel_size
        ),
        use_condition=True,
        use_time=True,
    )


# ================================================================
# BUILD CONDITIONED SPLINE DATALOADER
# ================================================================

def build_conditioned_spline_loader(
    train_q,
    train_b,
    train_time,
    *,
    args,
):

    B = train_q.shape[0]
    T = train_q.shape[1]

    # ------------------------------------------------------------
    # OPTIONAL TEMPORAL SUBSAMPLING
    # ------------------------------------------------------------

    if (
        float(
            args.partial_train_ratio
        )
        <
        1.0
    ):

        train_q, sampled_time = (
            load_partial_data(
                train_q,
                ratio=(
                    args.partial_train_ratio
                ),
                seed=(
                    args.partial_data_seed
                ),
            )
        )

        train_time = sampled_time

    retained_T = (
        train_q.shape[
            1
        ]
    )

    intervals_per_trajectory = (
        retained_T
        -
        1
    )

    # ------------------------------------------------------------
    # SPLINES
    # ------------------------------------------------------------

    if args.spline_type == "linear":

        (
            spline_coef,
            start_time,
            end_time,
        ) = linear_spline(
            train_q,
            time=train_time,
        )

    else:

        (
            spline_coef,
            start_time,
            end_time,
        ) = quintic_spline_batch(
            train_q,
            time=train_time,
            batch_size=(
                args.spline_batch_size
            ),
        )

    # ------------------------------------------------------------
    # CONDITION ALIGNMENT
    #
    # Spline flattening order:
    #
    # trajectory 0:
    #   interval 0
    #   interval 1
    #   ...
    #
    # trajectory 1:
    #   interval 0
    #   ...
    #
    # So each bathymetry must be repeated T-1 times.
    # ------------------------------------------------------------

    condition = np.repeat(
        train_b,
        intervals_per_trajectory,
        axis=0,
    )

    if (
        condition.shape[0]
        !=
        spline_coef.shape[0]
    ):

        raise RuntimeError(
            "Bathymetry / spline interval "
            "alignment failed.\n"
            f"spline_coef: {spline_coef.shape}\n"
            f"condition:   {condition.shape}"
        )

    print(
        "Conditioned spline dataloader ready."
    )

    print(
        "Spline intervals:",
        spline_coef.shape[
            0
        ],
    )

    print(
        "Intervals/trajectory:",
        intervals_per_trajectory,
    )

    return build_dataloader(
        spline_coef,
        t1=start_time,
        t2=end_time,
        c=condition,
        batch_size=(
            args.batch_size
        ),
        num_epochs=(
            args.epochs
            +
            5
        ),
        seed=args.seed,
    )


# ================================================================
# PREPARE ONE CFO BATCH
# ================================================================

def prepare_training_batch(
    raw_batch,
    *,
    time_key,
    noise_key,
):

    (
        spline_coef,
        t_start,
        t_end,
        condition,
        *_,
    ) = raw_batch

    spline_coef = (
        spline_coef[
            0
        ]
    )

    t_start = (
        t_start[
            0
        ]
    )

    t_end = (
        t_end[
            0
        ]
    )

    condition = (
        condition[
            0
        ]
    )

    dt = (
        t_end
        -
        t_start
    )

    delta_t = jax.random.uniform(
        time_key,
        shape=dt.shape,
        minval=0.0,
        maxval=dt,
    )

    x0 = spline_coef[
        :,
        0,
    ]

    eps = jax.random.normal(
        noise_key,
        x0.shape,
    )

    return (
        spline_coef,
        condition,
        t_start,
        t_end,
        delta_t,
        eps,
    )


# ================================================================
# EVALUATION
# ================================================================

def evaluate_rollout(
    method,
    state,
    split,
    *,
    steps_per_segment,
):

    q = np.asarray(
        split[
            "q"
        ],
        dtype=np.float32,
    )

    b = np.asarray(
        split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    q0 = jnp.asarray(
        q[
            :,
            0
        ]
    )

    condition = jnp.asarray(
        b
    )

    pred = method.uniform_inference(
        state,
        q0,
        trajectory_points_num=(
            q.shape[
                1
            ]
        ),
        steps_per_segment=(
            steps_per_segment
        ),
        condition=condition,
        method="RK4",
    )

    rel_l2 = float(
        relative_L2_error(
            q,
            pred,
        )
    )

    rmse_value = float(
        rmse(
            q,
            pred,
        )
    )

    rel_fro = float(
        relative_frobenius_error(
            q,
            pred,
        )
    )

    return (
        rel_l2,
        rmse_value,
        rel_fro,
        pred,
    )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    np.random.seed(
        args.seed
    )

    # ------------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------------

    lam_wb_name = lambda_name(
        args.lambda_wb
    )

    if args.ckpt_dir is None:

        ckpt_dir = (
            PROJECT_ROOT
            /
            "checkpoints"
            /
            "wb_bathy_bed_pi"
            /
            args.architecture
            /
            f"lamwb_{lam_wb_name}"
            /
            f"seed{args.seed}"
        )

    else:

        ckpt_dir = Path(
            args.ckpt_dir
        )

    if args.results_dir is None:

        results_dir = (
            PROJECT_ROOT
            /
            "results"
            /
            "wb_bathy_bed_pi"
            /
            args.architecture
            /
            f"lamwb_{lam_wb_name}"
            /
            f"seed{args.seed}"
        )

    else:

        results_dir = Path(
            args.results_dir
        )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------

    dataset_path = (
        PROJECT_ROOT
        /
        args.dataset_path
    )

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

    train_q = np.asarray(
        train_split[
            "q"
        ],
        dtype=np.float32,
    )

    train_b = np.asarray(
        train_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    train_time = (
        trajectory_time_array(
            train_split
        )
    )

    input_shape = (
        train_q.shape[
            2:
        ]
    )

    condition_shape = (
        train_b.shape[
            1:
        ]
    )

    print()
    print(
        "=" * 60
    )

    print(
        "WELL-BALANCED BED-PI-CFO"
    )

    print(
        "=" * 60
    )

    print(
        "Architecture:",
        args.architecture,
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "State shape:",
        input_shape,
    )

    print(
        "Condition shape:",
        condition_shape,
    )

    print(
        "Train/eval/test:",
        train_q.shape[0],
        eval_split["q"].shape[0],
        test_split["q"].shape[0],
    )

    print(
        "lambda_PDE:",
        args.lambda_pde,
    )

    print(
        "lambda_bed:",
        args.lambda_bed,
    )

    print(
        "lambda_WB:",
        args.lambda_wb,
    )

    print(
        "WB eta0:",
        args.wb_eta0,
    )

    print(
        "=" * 60
    )

    # ------------------------------------------------------------
    # DATA LOADER
    # ------------------------------------------------------------

    loader = (
        build_conditioned_spline_loader(
            train_q,
            train_b,
            train_time,
            args=args,
        )
    )

    # ------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------

    model = build_backbone(
        args,
        num_channels=(
            input_shape[
                -1
            ]
        ),
    )

    method = (
        WellBalancedBathymetryBedPICFO(
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
            lambda_wb=(
                args.lambda_wb
            ),
            wb_eta0=(
                args.wb_eta0
            ),
            dx=args.dx,
            dy=args.dy,
            gravity=(
                args.gravity
            ),
        )
    )

    # ------------------------------------------------------------
    # TRAIN STATE
    # ------------------------------------------------------------

    state = init_cfo_train_state(
        method,
        seed=args.seed,
        learning_rate=(
            args.lr
        ),
        beta1=args.beta1,
        beta2=args.beta2,
    )

    parameter_count = sum(
        x.size
        for x
        in tree_leaves(
            state.params
        )
    )

    print(
        "Model parameters:",
        parameter_count,
    )

    # ------------------------------------------------------------
    # LOSS FUNCTION
    # ------------------------------------------------------------

    def loss_with_aux(
        params,
        batch,
    ):

        components = (
            method.loss_components(
                params,
                batch,
            )
        )

        if len(
            components
        ) != 5:

            raise RuntimeError(
                "Expected loss_components() "
                "to return exactly:\n"
                "(total, CFO, PDE, bed, WB)\n"
                f"but got {len(components)} values."
            )

        (
            total,
            cfo,
            pde,
            bed,
            wb,
        ) = components

        return (
            total,
            (
                cfo,
                pde,
                bed,
                wb,
            ),
        )

    def train_step(
        state,
        batch,
    ):

        (
            (
                total,
                (
                    cfo,
                    pde,
                    bed,
                    wb,
                ),
            ),
            grads,
        ) = value_and_grad(
            loss_with_aux,
            has_aux=True,
        )(
            state.params,
            batch,
        )

        state = (
            state.apply_gradients(
                grads=grads
            )
        )

        return (
            state,
            total,
            cfo,
            pde,
            bed,
            wb,
        )

    train_step_jit = jax.jit(
        train_step
    )

    # ------------------------------------------------------------
    # TF -> JAX PIPELINE
    # ------------------------------------------------------------

    data = map(
        prepare_tf_data,
        loader,
    )

    data = (
        prefetch_to_device(
            data,
            2,
        )
    )

    # ------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------

    total_history = []
    cfo_history = []
    pde_history = []
    bed_history = []
    wb_history = []

    validation_epochs = []
    validation_l2 = []
    validation_rmse = []
    validation_fro = []

    best_state = state

    best_validation_l2 = float(
        "inf"
    )

    best_epoch = -1

    rng_key = (
        jax.random.PRNGKey(
            args.seed
        )
    )

    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------

    print()
    print(
        "Starting training..."
    )

    progress = trange(
        args.epochs,
        desc="Training",
    )

    for epoch in progress:

        (
            rng_key,
            time_key,
            noise_key,
        ) = jax.random.split(
            rng_key,
            3,
        )

        raw_batch = next(
            data
        )

        batch = (
            prepare_training_batch(
                raw_batch,
                time_key=time_key,
                noise_key=noise_key,
            )
        )

        (
            state,
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            wb_loss,
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

        wb_value = float(
            wb_loss
        )

        total_history.append(
            total_value
        )

        cfo_history.append(
            cfo_value
        )

        pde_history.append(
            pde_value
        )

        bed_history.append(
            bed_value
        )

        wb_history.append(
            wb_value
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

                "wb":
                    f"{wb_value:.3e}",
            }
        )

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        should_eval = (
            (
                epoch + 1
            )
            %
            args.eval_interval
            ==
            0
        )

        if should_eval:

            (
                current_l2,
                current_rmse,
                current_fro,
                _,
            ) = evaluate_rollout(
                method,
                state,
                eval_split,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

            validation_epochs.append(
                epoch + 1
            )

            validation_l2.append(
                current_l2
            )

            validation_rmse.append(
                current_rmse
            )

            validation_fro.append(
                current_fro
            )

            is_best = (
                current_l2
                <
                best_validation_l2
            )

            if is_best:

                best_validation_l2 = (
                    current_l2
                )

                best_epoch = (
                    epoch + 1
                )

                best_state = state

                save_train_state(
                    best_state,
                    str(
                        ckpt_dir
                    ),
                    prefix="best",
                    step=(
                        epoch + 1
                    ),
                    max_to_keep=1,
                )

            marker = (
                " [BEST]"
                if is_best
                else ""
            )

            print(
                f"\n[eval] "
                f"epoch={epoch + 1} "
                f"rel_l2={current_l2:.6f} "
                f"rmse={current_rmse:.6f} "
                f"rel_fro={current_fro:.6f}"
                f"{marker}"
            )

    # ------------------------------------------------------------
    # SAFETY:
    # If no validation occurred.
    # ------------------------------------------------------------

    if best_epoch < 0:

        best_state = state
        best_epoch = (
            args.epochs
        )

        (
            best_validation_l2,
            best_validation_rmse,
            best_validation_fro,
            _,
        ) = evaluate_rollout(
            method,
            best_state,
            eval_split,
            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        save_train_state(
            best_state,
            str(
                ckpt_dir
            ),
            prefix="best",
            step=best_epoch,
            max_to_keep=1,
        )

    # ------------------------------------------------------------
    # FINAL ID TEST — BEST VALIDATION MODEL
    # ------------------------------------------------------------

    (
        test_l2,
        test_rmse,
        test_fro,
        test_prediction,
    ) = evaluate_rollout(
        method,
        best_state,
        test_split,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------

    np.save(
        results_dir
        /
        "test_prediction.npy",
        np.asarray(
            test_prediction
        ),
    )

    np.savez(
        results_dir
        /
        "metrics.npz",

        architecture=np.asarray(
            args.architecture
        ),

        lambda_pde=np.asarray(
            args.lambda_pde
        ),

        lambda_bed=np.asarray(
            args.lambda_bed
        ),

        lambda_wb=np.asarray(
            args.lambda_wb
        ),

        best_epoch=np.asarray(
            best_epoch
        ),

        best_validation_l2=np.asarray(
            best_validation_l2
        ),

        test_relative_l2=np.asarray(
            test_l2
        ),

        test_rmse=np.asarray(
            test_rmse
        ),

        test_relative_frobenius=np.asarray(
            test_fro
        ),

        total_loss=np.asarray(
            total_history
        ),

        cfo_loss=np.asarray(
            cfo_history
        ),

        pde_loss=np.asarray(
            pde_history
        ),

        bed_loss=np.asarray(
            bed_history
        ),

        wb_loss=np.asarray(
            wb_history
        ),

        validation_epochs=np.asarray(
            validation_epochs
        ),

        validation_l2=np.asarray(
            validation_l2
        ),

        validation_rmse=np.asarray(
            validation_rmse
        ),

        validation_fro=np.asarray(
            validation_fro
        ),
    )

    # ------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL TRAINING LOSS COMPONENTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Total:         "
        f"{total_history[-1]:.6e}"
    )

    print(
        f"CFO:           "
        f"{cfo_history[-1]:.6e}"
    )

    print(
        f"PDE:           "
        f"{pde_history[-1]:.6e}"
    )

    print(
        f"Weighted PDE:  "
        f"{args.lambda_pde * pde_history[-1]:.6e}"
    )

    print(
        f"Bed:           "
        f"{bed_history[-1]:.6e}"
    )

    print(
        f"Weighted Bed:  "
        f"{args.lambda_bed * bed_history[-1]:.6e}"
    )

    print(
        f"WB:            "
        f"{wb_history[-1]:.6e}"
    )

    print(
        f"Weighted WB:   "
        f"{args.lambda_wb * wb_history[-1]:.6e}"
    )

    print()
    print(
        "=" * 60
    )

    print(
        "WB BED-PI-CFO ID TEST"
    )

    print(
        "=" * 60
    )

    print(
        "Architecture:",
        args.architecture,
    )

    print(
        "Relative L2:",
        f"{test_l2:.6f}",
    )

    print(
        "RMSE:",
        f"{test_rmse:.6f}",
    )

    print(
        "Rel Fro:",
        f"{test_fro:.6f}",
    )

    print(
        "Best validation epoch:",
        best_epoch,
    )

    print(
        "Best validation Rel L2:",
        f"{best_validation_l2:.6f}",
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        results_dir
    )

    print()
    print(
        "Best checkpoint saved to:"
    )

    print(
        ckpt_dir
        /
        "best"
    )


if __name__ == "__main__":
    main()