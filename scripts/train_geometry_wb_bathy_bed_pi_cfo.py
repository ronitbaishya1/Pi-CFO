"""
Train geometry-conditioned Well-Balanced Bed-PI-CFO models.

Available architectures
-----------------------

    fno
    geometry_fno
    geometry_ufno
    geometry_ffno

Physics objective
-----------------

L_total
=
L_CFO
+
lambda_PDE * L_PDE
+
lambda_bed * L_bed
+
lambda_WB * L_WB

The PI-CFO / Bed / WB formulation is unchanged.

Only the neural-operator backbone changes.
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


os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "3",
)

os.environ.setdefault(
    "ABSL_MIN_LOG_LEVEL",
    "3",
)


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


from models.fno import (
    FNO2d,
)

from models.geometry_fno import (
    GeometryFNO2d,
)

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from models.geometry_ffno import (
    GeometryFFNO2d,
)

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
    prepare_tf_data,
    prefetch_to_device,
)

from utils.checkpoints import (
    save_train_state,
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
            "Train geometry-conditioned "
            "WB-Bed-PI-CFO models."
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
    # ARCHITECTURE
    # -----------------------------------------------------------------

    parser.add_argument(
        "--architecture",
        type=str,
        default="geometry_fno",
        choices=[
            "fno",
            "geometry_fno",
            "geometry_ufno",
            "geometry_ffno",
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
        "--geometry-width",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--geometry-depth",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num-u-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ffno-expansion",
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
    # PHYSICS
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

    # Our selected WB value.
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

    # Physical domain is [-2.5,2.5]^2.
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

    # Optional manual override.
    parser.add_argument(
        "--dx",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--dy",
        type=float,
        default=None,
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

def build_backbone(
    args,
    *,
    num_channels,
    dx,
    dy,
):

    common = dict(
        num_channels=num_channels,
        modes1=args.modes1,
        modes2=args.modes2,
        width=args.width,
        num_blocks=args.num_blocks,
    )

    if args.architecture == "fno":

        return FNO2d(
            **common,
            use_condition=True,
            use_time=True,
        )

    if args.architecture == "geometry_fno":

        return GeometryFNO2d(
            **common,
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

    if args.architecture == "geometry_ufno":

        return GeometryUFNO2d(
            **common,
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

    if args.architecture == "geometry_ffno":

        return GeometryFFNO2d(
            **common,
            expansion=(
                args.ffno_expansion
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

    raise ValueError(
        f"Unsupported architecture: "
        f"{args.architecture}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    np.random.seed(
        args.seed
    )

    # -----------------------------------------------------------------
    # DATASET PATH
    # -----------------------------------------------------------------

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
            args.dx
        )
        if args.dx is not None
        else
        float(
            args.x_length
        )
        /
        nx
    )

    dy = (
        float(
            args.dy
        )
        if args.dy is not None
        else
        float(
            args.y_length
        )
        /
        ny
    )

    # -----------------------------------------------------------------
    # OUTPUT PATHS
    # -----------------------------------------------------------------

    lam_wb = lambda_name(
        args.lambda_wb
    )

    default_root = (
        PROJECT_ROOT
        /
        "checkpoints"
        /
        "geometry_wb_bed_pi"
        /
        args.architecture
        /
        f"res{nx}"
        /
        f"lamwb_{lam_wb}"
        /
        f"seed{args.seed}"
    )

    if args.ckpt_dir is None:

        ckpt_dir = (
            default_root.resolve()
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
            PROJECT_ROOT
            /
            "results"
            /
            "geometry_wb_bed_pi"
            /
            args.architecture
            /
            f"res{nx}"
            /
            f"lamwb_{lam_wb}"
            /
            f"seed{args.seed}"
        ).resolve()

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

    # -----------------------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "GEOMETRY-CONDITIONED "
        "WB-BED-PI-CFO"
    )

    print(
        "=" * 70
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
        "Resolution:",
        f"{nx} x {ny}",
    )

    print(
        "dx:",
        dx,
    )

    print(
        "dy:",
        dy,
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
        "=" * 70
    )

    # -----------------------------------------------------------------
    # SPLINE DATA
    # -----------------------------------------------------------------

    loader = (
        build_conditioned_spline_loader(
            train_q,
            train_b,
            train_time,
            args=args,
        )
    )

    # -----------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------

    model = build_backbone(
        args,
        num_channels=(
            input_shape[
                -1
            ]
        ),
        dx=dx,
        dy=dy,
    )

    # -----------------------------------------------------------------
    # SAME WB-BED-PI-CFO PHYSICS
    # -----------------------------------------------------------------

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

    print(
        "Model parameters:",
        f"{parameter_count:,}",
    )

    # -----------------------------------------------------------------
    # LOSS
    # -----------------------------------------------------------------

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
                "to return:\n"
                "(total, CFO, PDE, bed, WB)"
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

    # -----------------------------------------------------------------
    # DATA PIPELINE
    # -----------------------------------------------------------------

    data = map(
        prepare_tf_data,
        loader,
    )

    data = prefetch_to_device(
        data,
        2,
    )

    # -----------------------------------------------------------------
    # HISTORY
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------

    progress = trange(
        args.epochs,
        desc=(
            args.architecture
        ),
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
            total=f"{total_value:.3e}",
            cfo=f"{cfo_value:.3e}",
            pde=f"{pde_value:.3e}",
            bed=f"{bed_value:.3e}",
            wb=f"{wb_value:.3e}",
        )

        if (
            (epoch + 1)
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
                    step=best_epoch,
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

    # -----------------------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------------------

    if best_epoch < 0:

        best_state = state
        best_epoch = args.epochs

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
            step=best_epoch,
            max_to_keep=1,
        )

    # -----------------------------------------------------------------
    # TEST
    # -----------------------------------------------------------------

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

        resolution=np.asarray(
            nx
        ),

        dx=np.asarray(
            dx
        ),

        dy=np.asarray(
            dy
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

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Architecture:",
        args.architecture,
    )

    print(
        "Resolution:",
        f"{nx} x {ny}",
    )

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

    print(
        "Results:",
        results_dir,
    )

    print(
        "Checkpoint:",
        ckpt_dir
        /
        "best",
    )


if __name__ == "__main__":
    main()