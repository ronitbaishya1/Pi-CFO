"""
Short-horizon rollout-stability fine-tuning for the final
64x64 Geometry-U-FNO WB-Bed-PI-CFO model.

Established model
-----------------
Geometry-U-FNO with

    L_base
    =
    L_CFO
    + lambda_PDE * L_PDE
    + lambda_bed * L_bed
    + lambda_WB * L_WB

The established losses are NOT replaced.

New fine-tuning objective
-------------------------

    L_total
    =
    L_base
    + lambda_rollout * L_rollout
    + lambda_hf * L_HF

where L_rollout compares short differentiable RK4 rollouts against
the true trajectory.

The optional L_HF term penalizes only the HIGH-FREQUENCY PART OF
THE ROLLOUT ERROR. It does not smooth the prediction itself and does
not suppress legitimate physical high-frequency content.

Recommended first experiment:

    lambda_PDE     = 0.01
    lambda_bed     = 0.70
    lambda_WB      = 0.10

    lambda_rollout = 0.05
    rollout_horizon = 2
    rollout_substeps = 2

    lambda_hf = 0.0

First establish whether rollout consistency alone fixes the
recursive instability. Only activate L_HF later if necessary.
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
    load_train_state,
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
            "Fine-tune 64x64 Geometry-U-FNO using "
            "short-horizon differentiable rollout consistency."
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
            "swe_bathy_64_id.h5"
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
    # BASE CHECKPOINT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--base-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res64/"
            "lamwb_0p1/"
            "seed0"
        ),
    )

    parser.add_argument(
        "--base-prefix",
        type=str,
        default="best",
    )

    parser.add_argument(
        "--base-step",
        type=int,
        default=None,
        help=(
            "Checkpoint step. "
            "If omitted, latest checkpoint under base-prefix is used."
        ),
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
    # ESTABLISHED PHYSICS LOSSES
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.01,
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
    # ROLLOUT-STABILITY LOSS
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-rollout",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=2,
        help=(
            "Number of consecutive dataset intervals "
            "used by differentiable free rollout."
        ),
    )

    parser.add_argument(
        "--rollout-substeps",
        type=int,
        default=2,
        help=(
            "RK4 internal steps inside each dataset interval."
        ),
    )

    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # CHANNEL WEIGHTING
    #
    # More emphasis is placed on hu and hv because those are where
    # recursive high-frequency error is currently growing.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--weight-h",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--weight-hu",
        type=float,
        default=0.375,
    )

    parser.add_argument(
        "--weight-hv",
        type=float,
        default=0.375,
    )

    # -----------------------------------------------------------------
    # OPTIONAL HIGH-FREQUENCY ERROR REGULARIZATION
    #
    # Leave OFF for the first experiment.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-hf",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--hf-threshold",
        type=float,
        default=0.65,
        help=(
            "Normalized radial spatial-frequency threshold. "
            "Same interpretation as the rollout diagnostics."
        ),
    )

    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
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
        default=1.0e-5,
        help=(
            "Use a lower LR than original training because this is "
            "checkpoint fine-tuning."
        ),
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
    # VALIDATION
    # -----------------------------------------------------------------

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval-steps-per-segment",
        type=int,
        default=4,
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
# DIFFERENTIABLE RK4 WITH PARAMS
# =====================================================================

def rk4_interval(
    method,
    params,
    x,
    t_start,
    t_end,
    bathymetry,
    *,
    substeps,
):
    """
    Differentiably integrate one trajectory interval.

    x:
        (B,H,W,C)

    t_start, t_end:
        (B,)

    bathymetry:
        (B,H,W,1)
    """

    substeps = max(
        int(substeps),
        1,
    )

    dt = (
        t_end
        -
        t_start
    ) / float(
        substeps
    )

    dt_field = dt[
        :,
        None,
        None,
        None,
    ]

    current_t = t_start

    for _ in range(
        substeps
    ):

        k1 = method._model_apply(
            params,
            x,
            current_t,
            bathymetry,
        )

        k2 = method._model_apply(
            params,
            x
            +
            0.5
            *
            dt_field
            *
            k1,
            current_t
            +
            0.5
            *
            dt,
            bathymetry,
        )

        k3 = method._model_apply(
            params,
            x
            +
            0.5
            *
            dt_field
            *
            k2,
            current_t
            +
            0.5
            *
            dt,
            bathymetry,
        )

        k4 = method._model_apply(
            params,
            x
            +
            dt_field
            *
            k3,
            current_t
            +
            dt,
            bathymetry,
        )

        x = (
            x
            +
            (
                dt_field
                /
                6.0
            )
            *
            (
                k1
                +
                2.0
                *
                k2
                +
                2.0
                *
                k3
                +
                k4
            )
        )

        current_t = (
            current_t
            +
            dt
        )

    return x


# =====================================================================
# HIGH-PASS ERROR
# =====================================================================

def high_frequency_error_loss(
    normalized_error,
    *,
    threshold,
    channel_weights,
):
    """
    Penalize the high-frequency part of PREDICTION ERROR.

    We do NOT penalize the high-frequency content of the prediction
    itself. Therefore legitimate shocks/waves are not automatically
    suppressed.

    normalized_error:
        (B,H,W,C)
    """

    _, nx, ny, _ = (
        normalized_error.shape
    )

    fx = (
        jnp.fft.fftfreq(
            nx
        )
        /
        0.5
    )

    fy = (
        jnp.fft.fftfreq(
            ny
        )
        /
        0.5
    )

    radius = jnp.sqrt(
        fx[
            :,
            None
        ]
        ** 2
        +
        fy[
            None,
            :
        ]
        ** 2
    )

    mask = (
        radius
        >=
        float(
            threshold
        )
    ).astype(
        normalized_error.dtype
    )

    mask = mask[
        None,
        ...,
        None,
    ]

    error_fft = jnp.fft.fft2(
        normalized_error,
        axes=(
            1,
            2,
        ),
        norm="ortho",
    )

    high_fft = (
        error_fft
        *
        mask
    )

    high_error = (
        jnp.fft.ifft2(
            high_fft,
            axes=(
                1,
                2,
            ),
            norm="ortho",
        )
        .real
    )

    per_channel = jnp.mean(
        high_error**2,
        axis=(
            0,
            1,
            2,
        ),
    )

    return jnp.sum(
        channel_weights
        *
        per_channel
    )


# =====================================================================
# SHORT FREE-ROLLOUT LOSS
# =====================================================================

def short_rollout_loss(
    method,
    params,
    q0,
    targets,
    times,
    bathymetry,
    channel_scale,
    channel_weights,
    *,
    substeps,
    hf_threshold,
):
    """
    q0:
        (B,H,W,C)

    targets:
        (B,K,H,W,C)

    times:
        (B,K+1)

    Returns
    -------
    rollout_loss
    hf_loss
    """

    horizon = (
        targets.shape[
            1
        ]
    )

    current_q = q0

    scale = channel_scale[
        None,
        None,
        None,
        :
    ]

    rollout_total = jnp.asarray(
        0.0,
        dtype=q0.dtype,
    )

    hf_total = jnp.asarray(
        0.0,
        dtype=q0.dtype,
    )

    for k in range(
        horizon
    ):

        current_q = rk4_interval(
            method,
            params,
            current_q,
            times[
                :,
                k
            ],
            times[
                :,
                k
                +
                1
            ],
            bathymetry,
            substeps=substeps,
        )

        target = targets[
            :,
            k,
        ]

        normalized_error = (
            current_q
            -
            target
        ) / scale

        per_channel = jnp.mean(
            normalized_error**2,
            axis=(
                0,
                1,
                2,
            ),
        )

        rollout_total = (
            rollout_total
            +
            jnp.sum(
                channel_weights
                *
                per_channel
            )
        )

        hf_total = (
            hf_total
            +
            high_frequency_error_loss(
                normalized_error,
                threshold=(
                    hf_threshold
                ),
                channel_weights=(
                    channel_weights
                ),
            )
        )

    horizon_float = float(
        horizon
    )

    return (
        rollout_total
        /
        horizon_float,
        hf_total
        /
        horizon_float,
    )


# =====================================================================
# RANDOM SHORT-ROLLOUT BATCH
# =====================================================================

def sample_rollout_batch(
    rng,
    train_q,
    train_b,
    train_time,
    *,
    batch_size,
    horizon,
):
    """
    Randomly sample trajectory windows:

        q_i
        -> q_{i+1}
        -> ...
        -> q_{i+horizon}
    """

    n_trajectories = int(
        train_q.shape[
            0
        ]
    )

    n_times = int(
        train_q.shape[
            1
        ]
    )

    if horizon >= n_times:

        raise ValueError(
            "rollout_horizon must be smaller "
            "than the number of trajectory snapshots."
        )

    trajectory_ids = rng.integers(
        0,
        n_trajectories,
        size=batch_size,
    )

    max_start = (
        n_times
        -
        horizon
        -
        1
    )

    start_ids = rng.integers(
        0,
        max_start
        +
        1,
        size=batch_size,
    )

    q0_list = []

    target_list = []

    time_list = []

    bathy_list = []

    for trajectory_id, start_id in zip(
        trajectory_ids,
        start_ids,
    ):

        end_id = (
            start_id
            +
            horizon
        )

        q0_list.append(
            train_q[
                trajectory_id,
                start_id,
            ]
        )

        target_list.append(
            train_q[
                trajectory_id,
                start_id
                +
                1:
                end_id
                +
                1,
            ]
        )

        time_list.append(
            train_time[
                trajectory_id,
                start_id:
                end_id
                +
                1,
            ]
        )

        bathy_list.append(
            train_b[
                trajectory_id
            ]
        )

    return (
        jnp.asarray(
            np.stack(
                q0_list,
                axis=0,
            ),
            dtype=jnp.float32,
        ),
        jnp.asarray(
            np.stack(
                target_list,
                axis=0,
            ),
            dtype=jnp.float32,
        ),
        jnp.asarray(
            np.stack(
                time_list,
                axis=0,
            ),
            dtype=jnp.float32,
        ),
        jnp.asarray(
            np.stack(
                bathy_list,
                axis=0,
            ),
            dtype=jnp.float32,
        ),
    )


# =====================================================================
# CHANNEL METRICS
# =====================================================================

def channel_relative_l2(
    truth,
    prediction,
):
    """
    Relative L2 independently for h, hu and hv.
    """

    truth = np.asarray(
        truth,
        dtype=np.float64,
    )

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    values = []

    for channel in range(
        truth.shape[
            -1
        ]
    ):

        reference = truth[
            ...,
            channel
        ]

        estimate = prediction[
            ...,
            channel
        ]

        numerator = np.sqrt(
            np.sum(
                (
                    estimate
                    -
                    reference
                )
                ** 2
            )
        )

        denominator = np.sqrt(
            np.sum(
                reference**2
            )
        )

        values.append(
            float(
                numerator
                /
                max(
                    denominator,
                    1.0e-12,
                )
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    if args.lambda_rollout < 0.0:

        raise ValueError(
            "--lambda-rollout must be >= 0."
        )

    if args.lambda_hf < 0.0:

        raise ValueError(
            "--lambda-hf must be >= 0."
        )

    if args.rollout_horizon < 1:

        raise ValueError(
            "--rollout-horizon must be >= 1."
        )

    if args.rollout_substeps < 1:

        raise ValueError(
            "--rollout-substeps must be >= 1."
        )

    np.random.seed(
        args.seed
    )

    np_rng = np.random.default_rng(
        args.seed
        +
        1207
    )

    # =================================================================
    # DATA
    # =================================================================

    dataset_path = (
        Path(
            args.dataset_path
        )
        .expanduser()
    )

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

    # =================================================================
    # CHANNEL NORMALIZATION
    #
    # This is important because h ~ O(1), while hu/hv are smaller.
    # Without normalization, h could dominate rollout MSE.
    # =================================================================

    channel_scale = np.sqrt(
        np.mean(
            train_q**2,
            axis=(
                0,
                1,
                2,
                3,
            ),
        )
    ).astype(
        np.float32
    )

    channel_scale = np.maximum(
        channel_scale,
        1.0e-3,
    )

    channel_scale_jax = jnp.asarray(
        channel_scale,
        dtype=jnp.float32,
    )

    channel_weights_np = np.asarray(
        [
            args.weight_h,
            args.weight_hu,
            args.weight_hv,
        ],
        dtype=np.float32,
    )

    channel_weights_np = (
        channel_weights_np
        /
        np.sum(
            channel_weights_np
        )
    )

    channel_weights = jnp.asarray(
        channel_weights_np,
        dtype=jnp.float32,
    )

    # =================================================================
    # MODEL + ESTABLISHED METHOD
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

    # =================================================================
    # INITIALIZE TARGET STATE
    # =================================================================

    target_state = (
        init_cfo_train_state(
            method,
            seed=args.seed,
            learning_rate=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
        )
    )

    # =================================================================
    # RESTORE GOOD 64x64 CHECKPOINT
    # =================================================================

    base_ckpt_dir = (
        Path(
            args.base_ckpt_dir
        )
        .expanduser()
    )

    if not base_ckpt_dir.is_absolute():

        base_ckpt_dir = (
            PROJECT_ROOT
            /
            base_ckpt_dir
        )

    base_ckpt_dir = (
        base_ckpt_dir.resolve()
    )

    restored_state = (
        load_train_state(
            target_state,
            str(
                base_ckpt_dir
            ),
            prefix=(
                args.base_prefix
            ),
            step=(
                args.base_step
            ),
            max_to_keep=1,
        )
    )

    # =================================================================
    # RESET ADAM OPTIMIZER BUT KEEP RESTORED PARAMETERS
    #
    # Important for true low-LR fine-tuning.
    # =================================================================

    state = (
        init_cfo_train_state(
            method,
            seed=args.seed,
            learning_rate=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
        )
    )

    state = state.replace(
        params=(
            restored_state.params
        )
    )

    # =================================================================
    # OUTPUT DIRECTORIES
    # =================================================================

    rollout_name = (
        "K"
        +
        str(
            args.rollout_horizon
        )
        +
        "_lamroll_"
        +
        lambda_name(
            args.lambda_rollout
        )
        +
        "_lamhf_"
        +
        lambda_name(
            args.lambda_hf
        )
    )

    default_ckpt_dir = (
        PROJECT_ROOT
        /
        "checkpoints"
        /
        "rollout_stable"
        /
        "geometry_ufno"
        /
        f"res{nx}"
        /
        rollout_name
        /
        f"seed{args.seed}"
    )

    default_results_dir = (
        PROJECT_ROOT
        /
        "results"
        /
        "rollout_stable"
        /
        "geometry_ufno"
        /
        f"res{nx}"
        /
        rollout_name
        /
        f"seed{args.seed}"
    )

    ckpt_dir = (
        default_ckpt_dir
        if args.ckpt_dir is None
        else
        Path(
            args.ckpt_dir
        )
    )

    results_dir = (
        default_results_dir
        if args.results_dir is None
        else
        Path(
            args.results_dir
        )
    )

    ckpt_dir = (
        ckpt_dir
        .expanduser()
        .resolve()
    )

    results_dir = (
        results_dir
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
    # ESTABLISHED CFO SPLINE LOADER
    # =================================================================

    loader = (
        build_conditioned_spline_loader(
            train_q,
            train_b,
            train_time,
            args=args,
        )
    )

    data = map(
        prepare_tf_data,
        loader,
    )

    data = prefetch_to_device(
        data,
        2,
    )

    # =================================================================
    # BASELINE VALIDATION BEFORE FINE-TUNING
    # =================================================================

    (
        baseline_val_l2,
        baseline_val_rmse,
        baseline_val_fro,
        baseline_val_prediction,
    ) = evaluate_rollout(
        method,
        state,
        eval_split,
        steps_per_segment=(
            args.eval_steps_per_segment
        ),
    )

    baseline_val_channels = (
        channel_relative_l2(
            eval_split[
                "q"
            ],
            baseline_val_prediction,
        )
    )

    baseline_score = float(
        np.sum(
            channel_weights_np
            *
            baseline_val_channels
        )
    )

    # =================================================================
    # INFORMATION
    # =================================================================

    print()

    print(
        "=" * 76
    )

    print(
        "64x64 GEOMETRY-U-FNO ROLLOUT-STABILITY FINE-TUNING"
    )

    print(
        "=" * 76
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

    print()

    print(
        "Base checkpoint:",
        base_ckpt_dir,
    )

    print()

    print(
        "Established losses:"
    )

    print(
        "  lambda_PDE =",
        args.lambda_pde,
    )

    print(
        "  lambda_bed =",
        args.lambda_bed,
    )

    print(
        "  lambda_WB  =",
        args.lambda_wb,
    )

    print()

    print(
        "New rollout settings:"
    )

    print(
        "  lambda_rollout =",
        args.lambda_rollout,
    )

    print(
        "  rollout horizon =",
        args.rollout_horizon,
    )

    print(
        "  RK4 substeps/interval =",
        args.rollout_substeps,
    )

    print(
        "  lambda_HF =",
        args.lambda_hf,
    )

    print()

    print(
        "Channel scales [h, hu, hv]:",
        channel_scale,
    )

    print(
        "Channel weights [h, hu, hv]:",
        channel_weights_np,
    )

    print()

    print(
        "Baseline validation Rel-L2:",
        baseline_val_l2,
    )

    print(
        "Baseline validation channel Rel-L2:",
        baseline_val_channels,
    )

    print(
        "Baseline weighted rollout score:",
        baseline_score,
    )

    print(
        "=" * 76
    )

    # =================================================================
    # LOSS
    # =================================================================

    def loss_with_aux(
        params,
        base_batch,
        rollout_q0,
        rollout_targets,
        rollout_times,
        rollout_b,
    ):

        base_components = (
            method.loss_components(
                params,
                base_batch,
            )
        )

        base_total = (
            base_components[
                0
            ]
        )

        cfo_loss = (
            base_components[
                1
            ]
        )

        pde_loss = (
            base_components[
                2
            ]
        )

        bed_loss = (
            base_components[
                3
            ]
        )

        wb_loss = (
            base_components[
                4
            ]
        )

        (
            rollout_loss,
            hf_loss,
        ) = short_rollout_loss(
            method,
            params,
            rollout_q0,
            rollout_targets,
            rollout_times,
            rollout_b,
            channel_scale_jax,
            channel_weights,
            substeps=(
                args.rollout_substeps
            ),
            hf_threshold=(
                args.hf_threshold
            ),
        )

        total_loss = (
            base_total
            +
            args.lambda_rollout
            *
            rollout_loss
            +
            args.lambda_hf
            *
            hf_loss
        )

        return (
            total_loss,
            (
                base_total,
                cfo_loss,
                pde_loss,
                bed_loss,
                wb_loss,
                rollout_loss,
                hf_loss,
            ),
        )

    def train_step(
        state,
        base_batch,
        rollout_q0,
        rollout_targets,
        rollout_times,
        rollout_b,
    ):

        (
            (
                total,
                diagnostics,
            ),
            grads,
        ) = value_and_grad(
            loss_with_aux,
            has_aux=True,
        )(
            state.params,
            base_batch,
            rollout_q0,
            rollout_targets,
            rollout_times,
            rollout_b,
        )

        state = state.apply_gradients(
            grads=grads
        )

        return (
            state,
            total,
            diagnostics,
        )

    train_step_jit = jax.jit(
        train_step
    )

    # =================================================================
    # HISTORY
    # =================================================================

    total_history = []

    base_history = []

    cfo_history = []

    pde_history = []

    bed_history = []

    wb_history = []

    rollout_history = []

    hf_history = []

    validation_epochs = []

    validation_l2_history = []

    validation_score_history = []

    validation_h_history = []

    validation_hu_history = []

    validation_hv_history = []

    best_state = state

    best_score = (
        baseline_score
    )

    best_epoch = 0

    rng_key = jax.random.PRNGKey(
        args.seed
        +
        51
    )

    # =================================================================
    # TRAIN
    # =================================================================

    progress = trange(
        args.epochs,
        desc="rollout_finetune",
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

        # -------------------------------------------------------------
        # Existing CFO / PDE / bed / WB training batch
        # -------------------------------------------------------------

        raw_batch = next(
            data
        )

        base_batch = (
            prepare_training_batch(
                raw_batch,
                time_key=time_key,
                noise_key=noise_key,
            )
        )

        # -------------------------------------------------------------
        # Separate short free-rollout batch
        # -------------------------------------------------------------

        (
            rollout_q0,
            rollout_targets,
            rollout_times,
            rollout_b,
        ) = sample_rollout_batch(
            np_rng,
            train_q,
            train_b,
            train_time,
            batch_size=(
                args.rollout_batch_size
            ),
            horizon=(
                args.rollout_horizon
            ),
        )

        (
            state,
            total_loss,
            diagnostics,
        ) = train_step_jit(
            state,
            base_batch,
            rollout_q0,
            rollout_targets,
            rollout_times,
            rollout_b,
        )

        (
            base_loss,
            cfo_loss,
            pde_loss,
            bed_loss,
            wb_loss,
            rollout_loss,
            hf_loss,
        ) = diagnostics

        total_value = float(
            total_loss
        )

        base_value = float(
            base_loss
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

        rollout_value = float(
            rollout_loss
        )

        hf_value = float(
            hf_loss
        )

        total_history.append(
            total_value
        )

        base_history.append(
            base_value
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

        rollout_history.append(
            rollout_value
        )

        hf_history.append(
            hf_value
        )

        progress.set_postfix(
            total=(
                f"{total_value:.3e}"
            ),
            roll=(
                f"{rollout_value:.3e}"
            ),
            hf=(
                f"{hf_value:.3e}"
            ),
        )

        # -------------------------------------------------------------
        # VALIDATION
        # -------------------------------------------------------------

        should_eval = (
            (
                epoch
                +
                1
            )
            %
            args.eval_interval
            ==
            0
            or
            (
                epoch
                +
                1
            )
            ==
            args.epochs
        )

        if should_eval:

            (
                current_l2,
                current_rmse,
                current_fro,
                current_prediction,
            ) = evaluate_rollout(
                method,
                state,
                eval_split,
                steps_per_segment=(
                    args.eval_steps_per_segment
                ),
            )

            channel_l2 = (
                channel_relative_l2(
                    eval_split[
                        "q"
                    ],
                    current_prediction,
                )
            )

            score = float(
                np.sum(
                    channel_weights_np
                    *
                    channel_l2
                )
            )

            validation_epochs.append(
                epoch
                +
                1
            )

            validation_l2_history.append(
                current_l2
            )

            validation_score_history.append(
                score
            )

            validation_h_history.append(
                channel_l2[
                    0
                ]
            )

            validation_hu_history.append(
                channel_l2[
                    1
                ]
            )

            validation_hv_history.append(
                channel_l2[
                    2
                ]
            )

            is_best = (
                score
                <
                best_score
            )

            marker = (
                " <-- BEST"
                if is_best
                else
                ""
            )

            print()

            print(
                (
                    f"Epoch {epoch + 1:4d}: "
                    f"Val overall={current_l2:.6f} | "
                    f"h={channel_l2[0]:.6f} | "
                    f"hu={channel_l2[1]:.6f} | "
                    f"hv={channel_l2[2]:.6f} | "
                    f"score={score:.6f}"
                    f"{marker}"
                )
            )

            if is_best:

                best_score = (
                    score
                )

                best_epoch = (
                    epoch
                    +
                    1
                )

                best_state = (
                    state
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
    # IF NO FINE-TUNED EPOCH IMPROVED ON BASELINE
    # =================================================================

    if best_epoch == 0:

        print()

        print(
            "No fine-tuned checkpoint improved the "
            "baseline weighted validation score."
        )

        print(
            "Keeping restored baseline as the best state."
        )

        best_state = state.replace(
            params=(
                restored_state.params
            )
        )

    # =================================================================
    # TEST
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
            args.eval_steps_per_segment
        ),
    )

    test_channel_l2 = (
        channel_relative_l2(
            test_split[
                "q"
            ],
            test_prediction,
        )
    )

    test_score = float(
        np.sum(
            channel_weights_np
            *
            test_channel_l2
        )
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
        "training_summary.npz",

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

        lambda_rollout=np.asarray(
            args.lambda_rollout
        ),

        lambda_hf=np.asarray(
            args.lambda_hf
        ),

        rollout_horizon=np.asarray(
            args.rollout_horizon
        ),

        rollout_substeps=np.asarray(
            args.rollout_substeps
        ),

        channel_scale=np.asarray(
            channel_scale
        ),

        channel_weights=np.asarray(
            channel_weights_np
        ),

        baseline_validation_l2=np.asarray(
            baseline_val_l2
        ),

        baseline_validation_channels=np.asarray(
            baseline_val_channels
        ),

        baseline_validation_score=np.asarray(
            baseline_score
        ),

        best_epoch=np.asarray(
            best_epoch
        ),

        best_validation_score=np.asarray(
            best_score
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

        test_channel_relative_l2=np.asarray(
            test_channel_l2
        ),

        test_weighted_score=np.asarray(
            test_score
        ),

        total_loss=np.asarray(
            total_history
        ),

        base_loss=np.asarray(
            base_history
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

        rollout_loss=np.asarray(
            rollout_history
        ),

        high_frequency_loss=np.asarray(
            hf_history
        ),

        validation_epochs=np.asarray(
            validation_epochs
        ),

        validation_l2=np.asarray(
            validation_l2_history
        ),

        validation_score=np.asarray(
            validation_score_history
        ),

        validation_h=np.asarray(
            validation_h_history
        ),

        validation_hu=np.asarray(
            validation_hu_history
        ),

        validation_hv=np.asarray(
            validation_hv_history
        ),
    )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()

    print(
        "=" * 76
    )

    print(
        "ROLLOUT-STABILITY FINE-TUNING COMPLETE"
    )

    print(
        "=" * 76
    )

    print()

    print(
        "Baseline validation overall Rel-L2:",
        f"{baseline_val_l2:.6f}",
    )

    print(
        "Baseline validation channels:",
        baseline_val_channels,
    )

    print(
        "Baseline weighted score:",
        f"{baseline_score:.6f}",
    )

    print()

    print(
        "Best fine-tune epoch:",
        best_epoch,
    )

    print(
        "Best validation weighted score:",
        f"{best_score:.6f}",
    )

    print()

    print(
        "ID test overall Rel-L2:",
        f"{test_l2:.6f}",
    )

    print(
        "ID test channel Rel-L2 [h, hu, hv]:",
        test_channel_l2,
    )

    print(
        "ID test weighted score:",
        f"{test_score:.6f}",
    )

    print()

    print(
        "Results:",
        results_dir,
    )

    print(
        "Checkpoints:",
        ckpt_dir,
    )

    print(
        "=" * 76
    )


if __name__ == "__main__":

    main()