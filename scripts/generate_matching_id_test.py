"""
Generate matched SWE test trajectories at a different resolution,
different final time, or perturbed initial dam radius.

The RNG progression matches scripts/generate_swe_bathy.py:

    24 train
    6 eval
    6 test
    seed = 2026

Therefore the generated test terrains and baseline dam radii correspond
to the original ID test set when those defaults are preserved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))


from generate_swe_bathy import (
    build_grid,
    run_pyclaw,
)

from bathymetry import (
    make_bathymetry,
    parameter_vector,
    sample_id_parameters,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)

    parser.add_argument("--x-min", type=float, default=-2.5)
    parser.add_argument("--x-max", type=float, default=2.5)
    parser.add_argument("--y-min", type=float, default=-2.5)
    parser.add_argument("--y-max", type=float, default=2.5)

    parser.add_argument("--num-times", type=int, default=51)
    parser.add_argument("--tfinal", type=float, default=1.0)

    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--dry-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--sea-level", type=float, default=0.0)

    parser.add_argument("--eta-inside", type=float, default=2.0)
    parser.add_argument("--eta-outside", type=float, default=1.0)
    parser.add_argument("--lake-eta", type=float, default=2.0)
    parser.add_argument("--min-depth", type=float, default=0.20)

    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--eval-count", type=int, default=6)
    parser.add_argument("--test-count", type=int, default=6)

    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--dam-radius-min", type=float, default=0.30)
    parser.add_argument("--dam-radius-max", type=float, default=0.70)

    parser.add_argument(
        "--radius-offset",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    return parser.parse_args()


def obtain_original_test_cases(
    rng,
    *,
    train_count,
    eval_count,
    test_count,
    dam_radius_min,
    dam_radius_max,
):

    # ============================================================
    # TRAIN
    # ============================================================

    for _ in range(train_count):
        sample_id_parameters(rng)

    for _ in range(train_count):
        rng.uniform(
            dam_radius_min,
            dam_radius_max,
        )

    # ============================================================
    # EVAL
    # ============================================================

    for _ in range(eval_count):
        sample_id_parameters(rng)

    for _ in range(eval_count):
        rng.uniform(
            dam_radius_min,
            dam_radius_max,
        )

    # ============================================================
    # TEST TERRAIN
    # ============================================================

    test_parameters = [
        sample_id_parameters(rng)
        for _ in range(test_count)
    ]

    # Original write_split samples radii after terrain parameters.
    test_radii = [
        float(
            rng.uniform(
                dam_radius_min,
                dam_radius_max,
            )
        )
        for _ in range(test_count)
    ]

    return test_parameters, test_radii


def main():

    args = parse_args()

    output_path = Path(args.output).expanduser()

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path = output_path.resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        x,
        y,
        X,
        Y,
    ) = build_grid(
        args.nx,
        args.ny,
        args.x_min,
        args.x_max,
        args.y_min,
        args.y_max,
    )

    rng = np.random.default_rng(
        args.seed
    )

    (
        terrain_parameters,
        baseline_radii,
    ) = obtain_original_test_cases(
        rng,
        train_count=args.train_count,
        eval_count=args.eval_count,
        test_count=args.test_count,
        dam_radius_min=args.dam_radius_min,
        dam_radius_max=args.dam_radius_max,
    )

    q_all = []
    b_all = []
    params_all = []
    radii_all = []
    baseline_radii_all = []

    reference_time = None

    for index, (
        terrain_params,
        baseline_radius,
    ) in enumerate(
        zip(
            terrain_parameters,
            baseline_radii,
        )
    ):

        print()
        print("=" * 72)
        print(
            f"Matched test trajectory "
            f"{index + 1}/{args.test_count}"
        )
        print("=" * 72)

        bathymetry = make_bathymetry(
            X,
            Y,
            terrain_params,
        ).astype(
            np.float32
        )

        radius = (
            baseline_radius
            +
            args.radius_offset
        )

        if radius <= 0.0:
            raise ValueError(
                "Perturbed dam radius became non-positive."
            )

        print(
            f"baseline radius = {baseline_radius:.6f}"
        )

        print(
            f"used radius     = {radius:.6f}"
        )

        q, time = run_pyclaw(
            bathymetry=bathymetry,
            dam_radius=radius,
            num_times=args.num_times,
            tfinal=args.tfinal,
            g=args.g,
            dry_tolerance=args.dry_tolerance,
            sea_level=args.sea_level,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            eta_inside=args.eta_inside,
            eta_outside=args.eta_outside,
            min_depth=args.min_depth,
            lake_at_rest=False,
            lake_eta=args.lake_eta,
        )

        if reference_time is None:
            reference_time = np.asarray(
                time,
                dtype=np.float32,
            )

        elif not np.allclose(
            reference_time,
            time,
        ):
            raise RuntimeError(
                "Time grids differ between trajectories."
            )

        q_all.append(
            np.asarray(
                q,
                dtype=np.float32,
            )
        )

        b_all.append(
            bathymetry[
                ...,
                None
            ]
        )

        params_all.append(
            parameter_vector(
                terrain_params
            )
        )

        radii_all.append(
            radius
        )

        baseline_radii_all.append(
            baseline_radius
        )

    with h5py.File(
        output_path,
        "w",
    ) as h5:

        h5.attrs["nx"] = args.nx
        h5.attrs["ny"] = args.ny

        h5.attrs["x_min"] = args.x_min
        h5.attrs["x_max"] = args.x_max

        h5.attrs["y_min"] = args.y_min
        h5.attrs["y_max"] = args.y_max

        h5.attrs["gravity"] = args.g
        h5.attrs["seed"] = args.seed
        h5.attrs["tfinal"] = args.tfinal

        h5.attrs["radius_offset"] = (
            args.radius_offset
        )

        h5.create_dataset(
            "x",
            data=np.asarray(
                x,
                dtype=np.float32,
            ),
        )

        h5.create_dataset(
            "y",
            data=np.asarray(
                y,
                dtype=np.float32,
            ),
        )

        group = h5.create_group(
            "test_id"
        )

        group.create_dataset(
            "q",
            data=np.asarray(
                q_all,
                dtype=np.float32,
            ),
            compression="gzip",
        )

        group.create_dataset(
            "bathymetry",
            data=np.asarray(
                b_all,
                dtype=np.float32,
            ),
            compression="gzip",
        )

        group.create_dataset(
            "terrain_parameters",
            data=np.asarray(
                params_all,
                dtype=np.float32,
            ),
        )

        group.create_dataset(
            "baseline_dam_radius",
            data=np.asarray(
                baseline_radii_all,
                dtype=np.float32,
            ),
        )

        group.create_dataset(
            "dam_radius",
            data=np.asarray(
                radii_all,
                dtype=np.float32,
            ),
        )

        group.create_dataset(
            "time",
            data=reference_time,
        )

    print()
    print("=" * 72)
    print("MATCHED TEST DATASET COMPLETE")
    print("=" * 72)
    print(output_path)


if __name__ == "__main__":
    main()