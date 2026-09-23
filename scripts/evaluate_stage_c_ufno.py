"""
Unified Stage-C evaluator for Geometry-U-FNO.

Supports:
    baseline
    tan001
    tan005
    tan01

Works for:
    ID 32
    OOD 32
    perturbed initial conditions
    t > 1 long rollout
    zero-shot 32 -> 64
    OOD 64

IMPORTANT:
Actual stored dataset times are passed to CFO.infer_at().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.geometry_ufno import GeometryUFNO2d

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import init_cfo_train_state

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)

from utils.checkpoints import load_train_state


VARIANT_CHECKPOINTS = {

    "baseline":
        (
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "lamwb_0p1/"
            "seed0"
        ),

    "tan001":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p001/"
            "seed0"
        ),

    "tan005":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p005/"
            "seed0"
        ),

    "tan01":
        (
            "checkpoints/"
            "tangent_wb_bed_pi/"
            "geometry_ufno/"
            "res32/"
            "smooth_sigma1p5_lamtan_0p01/"
            "seed0"
        ),
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        required=True,
        choices=list(
            VARIANT_CHECKPOINTS.keys()
        ),
    )

    parser.add_argument(
        "--dataset-path",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="test_id",
    )

    parser.add_argument(
        "--checkpoint-dir",
        default=None,
    )

    parser.add_argument(
        "--checkpoint-prefix",
        default="best",
    )

    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
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

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def resolve_project_path(
    value,
):

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def read_grid(
    dataset_path,
):

    with h5py.File(
        dataset_path,
        "r",
    ) as h5:

        x = np.asarray(
            h5["x"],
            dtype=np.float64,
        )

        y = np.asarray(
            h5["y"],
            dtype=np.float64,
        )

    if x.size < 2 or y.size < 2:
        raise ValueError(
            "Invalid grid."
        )

    dx = float(
        np.mean(
            np.diff(x)
        )
    )

    dy = float(
        np.mean(
            np.diff(y)
        )
    )

    return x, y, dx, dy


def normalize_time_array(
    time,
):

    time = np.asarray(
        time,
        dtype=np.float64,
    )

    if time.ndim == 1:
        return time

    if time.ndim == 2:

        reference = time[0]

        if not np.allclose(
            time,
            reference[
                None,
                :
            ],
        ):
            raise ValueError(
                "Trajectories use different time grids."
            )

        return reference

    raise ValueError(
        f"Unexpected time shape: {time.shape}"
    )


def rollout_actual_times(
    method,
    state,
    q0,
    bathymetry,
    time,
    *,
    steps_per_segment,
):

    q = jnp.asarray(
        q0,
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        bathymetry,
        dtype=jnp.float32,
    )

    predictions = [
        np.asarray(q)
    ]

    internal_steps = max(
        int(
            steps_per_segment
        )
        +
        1,
        2,
    )

    for index in range(
        len(time)
        -
        1
    ):

        q = method.infer_at(
            state,
            q,
            s=float(
                time[index]
            ),
            t=float(
                time[
                    index
                    +
                    1
                ]
            ),
            steps=internal_steps,
            condition=condition,
            method="RK4",
        )

        predictions.append(
            np.asarray(q)
        )

    return np.stack(
        predictions,
        axis=1,
    )


def relative_l2(
    truth,
    prediction,
    eps=1.0e-12,
):

    numerator = np.linalg.norm(
        (
            prediction
            -
            truth
        ).ravel()
    )

    denominator = np.linalg.norm(
        truth.ravel()
    )

    return float(
        numerator
        /
        max(
            denominator,
            eps,
        )
    )


def rmse(
    truth,
    prediction,
):

    return float(
        np.sqrt(
            np.mean(
                (
                    prediction
                    -
                    truth
                )
                ** 2
            )
        )
    )


def relative_error_by_time(
    truth,
    prediction,
):

    values = []

    for i in range(
        truth.shape[1]
    ):

        values.append(
            relative_l2(
                truth[:, i],
                prediction[:, i],
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def main():

    args = parse_args()

    dataset_path = resolve_project_path(
        args.dataset_path
    )

    output_dir = resolve_project_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.checkpoint_dir is None:

        checkpoint_dir = resolve_project_path(
            VARIANT_CHECKPOINTS[
                args.variant
            ]
        )

    else:

        checkpoint_dir = resolve_project_path(
            args.checkpoint_dir
        )

    dataset = load_bathymetry_dataset(
        dataset_path
    )

    split = require_split(
        dataset,
        args.split,
    )

    truth = np.asarray(
        split["q"],
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        split["bathymetry"],
        dtype=np.float32,
    )

    time = normalize_time_array(
        split["time"]
    )

    (
        number_trajectories,
        number_times,
        nx,
        ny,
        channels,
    ) = truth.shape

    x, y, dx, dy = read_grid(
        dataset_path
    )

    print()
    print("=" * 72)
    print("STAGE-C EVALUATION")
    print("=" * 72)

    print("Variant:", args.variant)
    print("Dataset:", dataset_path)
    print("Split:", args.split)
    print("Resolution:", f"{nx} x {ny}")
    print(
        "Time:",
        f"{time[0]:.4f}",
        "->",
        f"{time[-1]:.4f}",
    )

    model = GeometryUFNO2d(
        num_channels=channels,

        modes1=args.modes1,
        modes2=args.modes2,

        width=args.width,

        num_blocks=args.num_blocks,
        num_u_blocks=args.num_u_blocks,

        geometry_width=args.geometry_width,
        geometry_depth=args.geometry_depth,

        dx=dx,
        dy=dy,

        include_gradient_magnitude=False,

        use_time=True,
    )

    method = (
        WellBalancedBathymetryBedPICFO(
            model=model,

            input_shape=(
                nx,
                ny,
                channels,
            ),

            condition_shape=(
                nx,
                ny,
                1,
            ),

            gamma=1.0e-5,

            spline_type="quintic",

            lambda_pde=0.03,
            lambda_bed=0.70,
            lambda_wb=0.10,

            wb_eta0=1.5,

            dx=dx,
            dy=dy,

            gravity=1.0,
        )
    )

    target_state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    state = load_train_state(
        target_state,
        ckpt_dir=str(
            checkpoint_dir
        ),
        prefix=args.checkpoint_prefix,
        step=args.checkpoint_step,
        max_to_keep=1,
    )

    prediction = rollout_actual_times(
        method,
        state,
        truth[:, 0],
        bathymetry,
        time,
        steps_per_segment=(
            args.steps_per_segment
        ),
    )

    global_l2 = relative_l2(
        truth,
        prediction,
    )

    global_rmse = rmse(
        truth,
        prediction,
    )

    h_l2 = relative_l2(
        truth[..., 0],
        prediction[..., 0],
    )

    hu_l2 = relative_l2(
        truth[..., 1],
        prediction[..., 1],
    )

    hv_l2 = relative_l2(
        truth[..., 2],
        prediction[..., 2],
    )

    momentum_l2 = relative_l2(
        truth[..., 1:3],
        prediction[..., 1:3],
    )

    b_time = (
        bathymetry[
            :,
            None,
            ...,
            0,
        ]
    )

    eta_truth = (
        truth[..., 0]
        +
        b_time
    )

    eta_prediction = (
        prediction[..., 0]
        +
        b_time
    )

    eta_l2 = relative_l2(
        eta_truth,
        eta_prediction,
    )

    error_by_time = (
        relative_error_by_time(
            truth,
            prediction,
        )
    )

    h_error_by_time = (
        relative_error_by_time(
            truth[..., 0:1],
            prediction[..., 0:1],
        )
    )

    momentum_error_by_time = (
        relative_error_by_time(
            truth[..., 1:3],
            prediction[..., 1:3],
        )
    )

    eta_error_by_time = (
        relative_error_by_time(
            eta_truth[
                ...,
                None
            ],
            eta_prediction[
                ...,
                None
            ],
        )
    )

    cell_area = dx * dy

    true_mass = (
        np.sum(
            truth[..., 0],
            axis=(2, 3),
        )
        *
        cell_area
    )

    predicted_mass = (
        np.sum(
            prediction[..., 0],
            axis=(2, 3),
        )
        *
        cell_area
    )

    mass_relative_error = np.mean(
        np.abs(
            predicted_mass
            -
            true_mass
        )
        /
        np.maximum(
            np.abs(
                true_mass
            ),
            1.0e-12,
        ),
        axis=0,
    )

    true_min_depth = np.min(
        truth[..., 0],
        axis=(
            0,
            2,
            3,
        ),
    )

    predicted_min_depth = np.min(
        prediction[..., 0],
        axis=(
            0,
            2,
            3,
        ),
    )

    np.save(
        output_dir
        /
        "prediction.npy",
        prediction,
    )

    np.savez(
        output_dir
        /
        "metrics.npz",

        variant=np.asarray(
            args.variant
        ),

        time=time,

        resolution_x=np.asarray(nx),
        resolution_y=np.asarray(ny),

        dx=np.asarray(dx),
        dy=np.asarray(dy),

        global_relative_l2=np.asarray(
            global_l2
        ),

        global_rmse=np.asarray(
            global_rmse
        ),

        h_relative_l2=np.asarray(
            h_l2
        ),

        hu_relative_l2=np.asarray(
            hu_l2
        ),

        hv_relative_l2=np.asarray(
            hv_l2
        ),

        momentum_relative_l2=np.asarray(
            momentum_l2
        ),

        eta_relative_l2=np.asarray(
            eta_l2
        ),

        error_by_time=(
            error_by_time
        ),

        h_error_by_time=(
            h_error_by_time
        ),

        momentum_error_by_time=(
            momentum_error_by_time
        ),

        eta_error_by_time=(
            eta_error_by_time
        ),

        true_mass=true_mass,

        predicted_mass=predicted_mass,

        mass_relative_error=(
            mass_relative_error
        ),

        true_min_depth=(
            true_min_depth
        ),

        predicted_min_depth=(
            predicted_min_depth
        ),
    )

    print()
    print("Global Rel-L2:", f"{global_l2:.8f}")
    print("RMSE:", f"{global_rmse:.8f}")

    print("h Rel-L2:", f"{h_l2:.8f}")
    print("hu Rel-L2:", f"{hu_l2:.8f}")
    print("hv Rel-L2:", f"{hv_l2:.8f}")

    print(
        "Momentum Rel-L2:",
        f"{momentum_l2:.8f}",
    )

    print(
        "eta Rel-L2:",
        f"{eta_l2:.8f}",
    )

    print(
        "Final-time Rel-L2:",
        f"{error_by_time[-1]:.8f}",
    )

    print(
        "Final mass error:",
        f"{mass_relative_error[-1]:.8e}",
    )

    print(
        "Minimum predicted depth:",
        f"{np.min(predicted_min_depth):.8f}",
    )

    print()
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()