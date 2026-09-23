"""
Zero-shot cross-resolution evaluation.

Evaluate a Geometry-U-FNO trained at 32x32 directly on a
64x64 PyClaw shallow-water dataset.

No retraining is performed.

The network parameters are restored from the 32x32 checkpoint, while
the geometry derivatives and SWE physics use the target 64x64 spacing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


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

from scripts.train_geometry_wb_bathy_bed_pi_cfo import (
    build_backbone,
)

from scripts.train_wb_bathy_bed_pi_cfo import (
    evaluate_rollout,
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

from utils.checkpoints import (
    load_train_state,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot 32-to-64 Geometry-U-FNO evaluation."
        )
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_64_id.h5"
        ),
    )

    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "lamwb_0p1/"
            "seed0"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "cross_resolution/"
            "geometry_ufno/"
            "32_to_64"
        ),
    )

    # -----------------------------------------------------------------
    # ARCHITECTURE
    # -----------------------------------------------------------------

    parser.add_argument(
        "--architecture",
        type=str,
        default="geometry_ufno",
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

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0e-5,
    )

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

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


# =====================================================================
# PATH
# =====================================================================

def resolve_path(
    value,
):

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    dataset_path = resolve_path(
        args.dataset_path
    )

    checkpoint_root = resolve_path(
        args.checkpoint_root
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # LOAD 64x64 DATA
    # -----------------------------------------------------------------

    dataset = load_bathymetry_dataset(
        dataset_path
    )

    eval_split = require_split(
        dataset,
        "eval",
    )

    test_split = require_split(
        dataset,
        "test_id",
    )

    q = np.asarray(
        test_split[
            "q"
        ],
        dtype=np.float32,
    )

    b = np.asarray(
        test_split[
            "bathymetry"
        ],
        dtype=np.float32,
    )

    input_shape = (
        q.shape[
            2:
        ]
    )

    condition_shape = (
        b.shape[
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
        args.x_length
        /
        nx
    )

    dy = (
        args.y_length
        /
        ny
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ZERO-SHOT CROSS-RESOLUTION TEST"
    )

    print(
        "=" * 72
    )

    print(
        "Training resolution: 32 x 32"
    )

    print(
        "Target resolution:",
        f"{nx} x {ny}",
    )

    print(
        "Target dx:",
        dx,
    )

    print(
        "Target dy:",
        dy,
    )

    # -----------------------------------------------------------------
    # BUILD THE SAME ARCHITECTURE AT TARGET RESOLUTION
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

    method = (
        WellBalancedBathymetryBedPICFO(
            model=model,

            input_shape=input_shape,

            condition_shape=(
                condition_shape
            ),

            gamma=args.gamma,

            spline_type="quintic",

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

    # -----------------------------------------------------------------
    # INITIALIZE TARGET-RESOLUTION STATE
    # -----------------------------------------------------------------

    state = init_cfo_train_state(
        method,
        seed=args.seed,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    # -----------------------------------------------------------------
    # RESTORE 32x32 PARAMETERS
    #
    # Parameter tensors are independent of H/W.
    # -----------------------------------------------------------------

    state = load_train_state(
        state,

        ckpt_dir=str(
            checkpoint_root
        ),

        prefix="best",

        step=None,

        max_to_keep=1,
    )

    print(
        "32x32 checkpoint restored successfully."
    )

    # -----------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------

    (
        eval_l2,
        eval_rmse,
        eval_fro,
        _,
    ) = evaluate_rollout(
        method,
        state,
        eval_split,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    (
        test_l2,
        test_rmse,
        test_fro,
        prediction,
    ) = evaluate_rollout(
        method,
        state,
        test_split,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    # -----------------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------------

    np.save(
        output_dir
        /
        "test_prediction.npy",

        np.asarray(
            prediction
        ),
    )

    np.savez(
        output_dir
        /
        "metrics.npz",

        source_resolution=np.asarray(
            32
        ),

        target_resolution=np.asarray(
            nx
        ),

        dx=np.asarray(
            dx
        ),

        dy=np.asarray(
            dy
        ),

        validation_relative_l2=np.asarray(
            eval_l2
        ),

        validation_rmse=np.asarray(
            eval_rmse
        ),

        validation_relative_fro=np.asarray(
            eval_fro
        ),

        test_relative_l2=np.asarray(
            test_l2
        ),

        test_rmse=np.asarray(
            test_rmse
        ),

        test_relative_fro=np.asarray(
            test_fro
        ),
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "ZERO-SHOT 32 -> 64 RESULT"
    )

    print(
        "=" * 72
    )

    print(
        "Validation Rel L2:",
        f"{eval_l2:.6f}",
    )

    print(
        "Test Rel L2:",
        f"{test_l2:.6f}",
    )

    print(
        "Test RMSE:",
        f"{test_rmse:.6f}",
    )

    print(
        "Test Rel Fro:",
        f"{test_fro:.6f}",
    )

    print(
        "Saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()