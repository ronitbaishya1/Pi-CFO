"""
Train smooth-tangent-consistent Geometry-U-FNO WB-Bed-PI-CFO.

Architecture
------------

    Geometry-U-FNO

Established objective
---------------------

    L_base
    =
    L_CFO
    +
    lambda_PDE * L_PDE
    +
    lambda_bed * L_bed
    +
    lambda_WB * L_WB

Stage-B objective
-----------------

    L_total
    =
    L_base
    +
    lambda_tangent * L_tangent

where

    L_tangent

matches the directional derivative of the neural vector field with
the directional derivative of the existing discrete well-balanced
SWE vector field.

Perturbations
-------------

Stage B uses smooth Gaussian-filtered perturbations because Stage A
showed that they provide a much more physically relevant tangent
diagnostic than grid-scale white noise.
"""

from __future__ import annotations

import argparse
import os
import sys

from pathlib import Path

import jax
import numpy as np

from jax import value_and_grad
from jax.tree_util import tree_leaves

from tqdm.auto import trange


# =====================================================================
# ENVIRONMENT
# =====================================================================

os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "3",
)

os.environ.setdefault(
    "ABSL_MIN_LOG_LEVEL",
    "3",
)


# =====================================================================
# PROJECT ROOT
# =====================================================================

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


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from tangent_wb_bathy_bed_pi_cfo import (
    TangentConsistentWellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.data import (
    prepare_tf_data,
    prefetch_to_device,
)

from utils.checkpoints import (
    save_train_state,
)

from utils.tangent_swe_bathy import (
    smooth_tangent_direction,
)

from scripts.train_wb_bathy_bed_pi_cfo import (
    lambda_name,
    trajectory_time_array,
    build_conditioned_spline_loader,
    prepare_training_batch,
    evaluate_rollout,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train smooth-tangent-consistent "
            "Geometry-U-FNO WB-Bed-PI-CFO."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # GEOMETRY-U-FNO
    # -----------------------------------------------------------------

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
        "--num-u-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--geometry-width",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--geometry-depth",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

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
        default=1.0e-4,
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
        default=1.0e-5,
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

    # -----------------------------------------------------------------
    # ESTABLISHED PHYSICS LOSSES
    # -----------------------------------------------------------------

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
        default=0.10,
    )

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--gravity",
        type=float,
        default=1.0,
    )

    # -----------------------------------------------------------------
    # STAGE-B TANGENT LOSS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-tangent",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--smooth-kernel-size",
        type=int,
        default=7,
    )

    # -----------------------------------------------------------------
    # DOMAIN
    # -----------------------------------------------------------------

    parser.add_argument(
        "--x-length",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--y-length",
        type=float,
        default=5.0,
    )

    # -----------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

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


# =====================================================================
# MODEL
# =====================================================================

def build_model(
    args,
    *,
    num_channels,
    dx,
    dy,
):

    return GeometryUFNO2d(
        num_channels=num_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_blocks=args.num_blocks,
        num_u_blocks=(
            args.num_u_blocks
        ),
        geometry_width=(
            args.geometry_width
        ),
        geometry_depth=(
            args.geometry_depth
        ),
        dx=dx,
        dy=dy,
        use_time=True,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------

    if args.lambda_tangent < 0.0:

        raise ValueError(
            "--lambda-tangent must be >= 0."
        )

    if args.smooth_sigma <= 0.0:

        raise ValueError(
            "--smooth-sigma must be > 0."
        )

    if (
        args.smooth_kernel_size
        %
        2
        ==
        0
    ):

        raise ValueError(
            "--smooth-kernel-size must be odd."
        )

    np.random.seed(
        args.seed
    )

    # =================================================================
    # DATA
    # =================================================================

    dataset_path = Path(
        args.dataset_path
    ).expanduser()

    if not dataset_path.is_absolute():

        dataset_path = (
            PROJECT_ROOT
            /
            dataset_path
        )

    dataset_path = (
        dataset_path.resolve()
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

    nx = int(
        input_shape[
            0
        ]
    )

    ny = int(
        input_shape[
            1
        ]
    )

    dx = (
        float(
            args.x_length
        )
        /
        nx
    )

    dy = (
        float(
            args.y_length
        )
        /
        ny
    )

    # =================================================================
    # UNIQUE OUTPUT DIRECTORIES
    #
    # Avoid checkpoint collisions between tangent weights.
    # =================================================================

    tangent_name = lambda_name(
        args.lambda_tangent
    )

    experiment_name = (
        "smooth_sigma"
        +
        lambda_name(
            args.smooth_sigma
        )
        +
        "_lamtan_"
        +
        tangent_name
    )

    default_ckpt_dir = (
        PROJECT_ROOT
        /
        "checkpoints"
        /
        "tangent_wb_bed_pi"
        /
        "geometry_ufno"
        /
        f"res{nx}"
        /
        experiment_name
        /
        f"seed{args.seed}"
    )

    default_results_dir = (
        PROJECT_ROOT
        /
        "results"
        /
        "tangent_wb_bed_pi"
        /
        "geometry_ufno"
        /
        f"res{nx}"
        /
        experiment_name
        /
        f"seed{args.seed}"
    )

    if args.ckpt_dir is None:

        ckpt_dir = (
            default_ckpt_dir.resolve()
        )

    else:

        ckpt_dir = (
            Path(
                args.ckpt_dir
            )
            .expanduser()
            .resolve()
        )

    if args.results_dir is None:

        results_dir = (
            default_results_dir.resolve()
        )

    else:

        results_dir = (
            Path(
                args.results_dir
            )
            .expanduser()
            .resolve()
        )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # SPLINE DATALOADER
    # =================================================================

    loader = (
        build_conditioned_spline_loader(
            train_q,
            train_b,
            train_time,
            args=args,
        )
    )

    # =================================================================
    # MODEL
    # =================================================================

    model = build_model(
        args,
        num_channels=(
            input_shape[
                -1
            ]
        ),
        dx=dx,
        dy=dy,
    )

    method = (
        TangentConsistentWellBalancedBathymetryBedPICFO(
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
            lambda_tangent=(
                args.lambda_tangent
            ),
            wb_eta0=(
                args.wb_eta0
            ),
            dx=dx,
            dy=dy,
            gravity=(
                args.gravity
            ),
        )
    )

    state = init_cfo_train_state(
        method,
        seed=args.seed,
        learning_rate=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
    )

    parameter_count = sum(
        leaf.size
        for leaf
        in tree_leaves(
            state.params
        )
    )

    # =================================================================
    # INFORMATION
    # =================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "SMOOTH-TANGENT GEOMETRY-U-FNO WB-BED-PI-CFO"
    )

    print(
        "=" * 72
    )

    print(
        "Dataset:",
        dataset_path,
    )

    print(
        "Resolution:",
        f"{nx} x {ny}",
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    print()

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
        "lambda_tangent:",
        args.lambda_tangent,
    )

    print()

    print(
        "Smooth sigma:",
        args.smooth_sigma,
        "cells",
    )

    print(
        "Smooth kernel:",
        (
            f"{args.smooth_kernel_size}"
            " x "
            f"{args.smooth_kernel_size}"
        ),
    )

    print(
        "=" * 72
    )

    # =================================================================
    # LOSS
    # =================================================================

    def loss_with_aux(
        params,
        batch,
    ):

        (
            total,
            cfo,
            pde,
            bed,
            wb,
            tangent,
            tangent_cosine,
            tangent_gain_ratio,
        ) = method.loss_components(
            params,
            batch,
        )

        return (
            total,
            (
                cfo,
                pde,
                bed,
                wb,
                tangent,
                tangent_cosine,
                tangent_gain_ratio,
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
                    tangent,
                    tangent_cosine,
                    tangent_gain_ratio,
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

        state = state.apply_gradients(
            grads=grads
        )

        return (
            state,
            total,
            cfo,
            pde,
            bed,
            wb,
            tangent,
            tangent_cosine,
            tangent_gain_ratio,
        )

    train_step_jit = jax.jit(
        train_step
    )

    # =================================================================
    # DATA PIPELINE
    # =================================================================

    data = map(
        prepare_tf_data,
        loader,
    )

    data = prefetch_to_device(
        data,
        2,
    )

    # =================================================================
    # HISTORY
    # =================================================================

    total_history = []

    cfo_history = []

    pde_history = []

    bed_history = []

    wb_history = []

    tangent_history = []

    tangent_cosine_history = []

    tangent_gain_ratio_history = []

    validation_epochs = []

    validation_l2 = []

    validation_rmse = []

    validation_fro = []

    best_state = state

    best_validation_l2 = float(
        "inf"
    )

    best_epoch = -1

    rng_key = jax.random.PRNGKey(
        args.seed
    )

    # =================================================================
    # TRAIN
    # =================================================================

    progress = trange(
        args.epochs,
        desc="tangent_geometry_ufno",
    )

    for epoch in progress:

        (
            rng_key,
            time_key,
            cfo_noise_key,
            tangent_noise_key,
        ) = jax.random.split(
            rng_key,
            4,
        )

        raw_batch = next(
            data
        )

        # -------------------------------------------------------------
        # Existing 6-element CFO/WB batch.
        # -------------------------------------------------------------

        base_batch = (
            prepare_training_batch(
                raw_batch,
                time_key=time_key,
                noise_key=cfo_noise_key,
            )
        )

        spline_coef = (
            base_batch[
                0
            ]
        )

        # -------------------------------------------------------------
        # State shape:
        #
        # (B,H,W,3)
        #
        # spline_coef is:
        #
        # (B,6,H,W,3)
        #
        # for quintic splines.
        # -------------------------------------------------------------

        tangent_shape = (
            spline_coef[
                :,
                0,
            ]
            .shape
        )

        # -------------------------------------------------------------
        # RAW GAUSSIAN FIELD
        # -------------------------------------------------------------

        raw_tangent_noise = (
            jax.random.normal(
                tangent_noise_key,
                tangent_shape,
                dtype=(
                    spline_coef.dtype
                ),
            )
        )

        # -------------------------------------------------------------
        # SMOOTH PHYSICAL PERTURBATION
        #
        # Same construction used during Stage-A smooth diagnostic.
        # -------------------------------------------------------------

        tangent_direction = (
            smooth_tangent_direction(
                raw_tangent_noise,
                sigma=(
                    args.smooth_sigma
                ),
                kernel_size=(
                    args.smooth_kernel_size
                ),
            )
        )

        # -------------------------------------------------------------
        # Append tangent direction to the established CFO batch.
        # -------------------------------------------------------------

        batch = (
            *base_batch,
            tangent_direction,
        )

        (
            state,
            total_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            wb_loss,
            tangent_loss,
            tangent_cosine,
            tangent_gain_ratio,
        ) = train_step_jit(
            state,
            batch,
        )

        # =============================================================
        # PYTHON VALUES
        # =============================================================

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

        tangent_value = float(
            tangent_loss
        )

        tangent_cosine_value = float(
            tangent_cosine
        )

        tangent_gain_ratio_value = float(
            tangent_gain_ratio
        )

        weighted_tangent_value = (
            args.lambda_tangent
            *
            tangent_value
        )

        # =============================================================
        # HISTORY
        # =============================================================

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

        tangent_history.append(
            tangent_value
        )

        tangent_cosine_history.append(
            tangent_cosine_value
        )

        tangent_gain_ratio_history.append(
            tangent_gain_ratio_value
        )

        # =============================================================
        # PROGRESS
        # =============================================================

        progress.set_postfix(
            total=(
                f"{total_value:.3e}"
            ),
            cfo=(
                f"{cfo_value:.3e}"
            ),
            pde=(
                f"{pde_value:.3e}"
            ),
            bed=(
                f"{bed_value:.3e}"
            ),
            wb=(
                f"{wb_value:.3e}"
            ),
            tan=(
                f"{tangent_value:.3e}"
            ),
            tan_w=(
                f"{weighted_tangent_value:.3e}"
            ),
            cos=(
                f"{tangent_cosine_value:.3f}"
            ),
            gain=(
                f"{tangent_gain_ratio_value:.3f}"
            ),
        )

        # =============================================================
        # VALIDATION
        # =============================================================

        if (
            (
                epoch
                +
                1
            )
            %
            args.eval_interval
            ==
            0
        ):

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
                epoch
                +
                1
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
                    epoch
                    +
                    1
                )

                best_state = state

                save_train_state(
                    best_state,
                    str(
                        ckpt_dir
                    ),
                    prefix="best",
                    step=(
                        best_epoch
                    ),
                    max_to_keep=1,
                )

            marker = (
                " [BEST]"
                if is_best
                else
                ""
            )

            print(
                f"\n[eval] "
                f"epoch={epoch + 1} "
                f"RelL2={current_l2:.6f} "
                f"RMSE={current_rmse:.6f} "
                f"RelFro={current_fro:.6f}"
                f"{marker}"
            )

    # =================================================================
    # FALLBACK
    # =================================================================

    if best_epoch < 0:

        best_state = state

        best_epoch = (
            args.epochs
        )

        (
            best_validation_l2,
            _,
            _,
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
            step=(
                best_epoch
            ),
            max_to_keep=1,
        )

    # =================================================================
    # FINAL ID TEST — BEST VALIDATION CHECKPOINT
    # =================================================================

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

    # =================================================================
    # SAVE
    # =================================================================

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
            "geometry_ufno"
        ),

        perturbation=np.asarray(
            "smooth"
        ),

        smooth_sigma=np.asarray(
            args.smooth_sigma
        ),

        smooth_kernel_size=np.asarray(
            args.smooth_kernel_size
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

        lambda_tangent=np.asarray(
            args.lambda_tangent
        ),

        parameter_count=np.asarray(
            parameter_count
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

        tangent_loss=np.asarray(
            tangent_history
        ),

        tangent_cosine=np.asarray(
            tangent_cosine_history
        ),

        tangent_gain_ratio=np.asarray(
            tangent_gain_ratio_history
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

    # =================================================================
    # FINAL REPORT
    # =================================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FINAL STAGE-B RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "Architecture:",
        "geometry_ufno",
    )

    print(
        "Perturbation:",
        "smooth",
    )

    print(
        "lambda_tangent:",
        args.lambda_tangent,
    )

    print()

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation Rel L2:",
        f"{best_validation_l2:.6f}",
    )

    print(
        "ID test Rel L2:",
        f"{test_l2:.6f}",
    )

    print(
        "ID test RMSE:",
        f"{test_rmse:.6f}",
    )

    print(
        "ID test Rel Fro:",
        f"{test_fro:.6f}",
    )

    print()

    print(
        "Final tangent loss:",
        f"{tangent_history[-1]:.6f}",
    )

    print(
        "Final tangent cosine:",
        f"{tangent_cosine_history[-1]:.6f}",
    )

    print(
        "Final tangent gain ratio:",
        f"{tangent_gain_ratio_history[-1]:.6f}",
    )

    print()

    print(
        "Results:",
        results_dir,
    )

    print(
        "Checkpoint:",
        ckpt_dir,
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":

    main()