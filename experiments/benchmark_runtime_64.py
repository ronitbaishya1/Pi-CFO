"""
Runtime benchmark:

    PyClaw
    Bathy-CFO
    Final Geometry-U-FNO

PyClaw should be run in the swegen environment.
Neural models should be run in the cfo environment.

Compilation/loading is excluded from neural-network timing by using
a warm-up rollout before measured repetitions.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=[
            "pyclaw",
            "models",
            "plot",
        ],
        required=True,
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/final_resolution_study/"
            "bathy_cfo/res64/seed0/best"
        ),
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    parser.add_argument(
        "--final-ckpt",
        type=str,
        default=(
            "checkpoints/rollout_stable/"
            "geometry_ufno/res64/"
            "K2_lamroll_0p1_lamhf_0/"
            "seed0/best"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/runtime_analysis/res64",
    )

    return parser.parse_args()


def resolve(path):

    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def output_paths(args):

    output_dir = resolve(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir,
        output_dir / "runtime_results.csv",
    )


def load_existing(csv_path):

    if not csv_path.exists():
        return []

    with open(
        csv_path,
        "r",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def save_rows(csv_path, new_rows):

    old_rows = load_existing(
        csv_path
    )

    names_to_replace = {
        row["model"]
        for row in new_rows
    }

    rows = [
        row
        for row in old_rows
        if row.get("model")
        not in names_to_replace
    ]

    rows.extend(
        new_rows
    )

    columns = [
        "model",
        "resolution",
        "snapshots",
        "rk4_steps_per_segment",
        "repeats",
        "mean_seconds",
        "std_seconds",
        "median_seconds",
        "min_seconds",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(row)


def timing_row(
    name,
    values,
    *,
    resolution,
    snapshots,
    rk4_steps,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "model":
            name,

        "resolution":
            resolution,

        "snapshots":
            snapshots,

        "rk4_steps_per_segment":
            rk4_steps,

        "repeats":
            len(values),

        "mean_seconds":
            float(np.mean(values)),

        "std_seconds":
            float(np.std(values)),

        "median_seconds":
            float(np.median(values)),

        "min_seconds":
            float(np.min(values)),
    }


# ================================================================
# PYCLAW
# ================================================================

def benchmark_pyclaw(args):

    from scripts.generate_gaussian_counterfactual import (
        simulate_case,
    )

    resolution = int(
        args.resolution
    )

    kwargs = dict(
        case_name="hill_center",
        amplitude=0.20,
        xc=0.0,
        yc=0.0,
        sigma=0.50,
        nx=resolution,
        ny=resolution,
        xlower=-2.5,
        xupper=2.5,
        ylower=-2.5,
        yupper=2.5,
        tfinal=1.0,
        num_snapshots=51,
        gravity=1.0,
        dam_radius=0.50,
    )

    # Warm-up.
    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        simulate_case(
            **kwargs
        )

    values = []

    for repeat in range(
        args.repeats
    ):

        start = time.perf_counter()

        with contextlib.redirect_stdout(
            io.StringIO()
        ):

            result = simulate_case(
                **kwargs
            )

        elapsed = (
            time.perf_counter()
            -
            start
        )

        values.append(
            elapsed
        )

        print(
            f"PyClaw run {repeat + 1}: "
            f"{elapsed:.6f} s"
        )

    return [
        timing_row(
            "PyClaw",
            values,
            resolution=resolution,
            snapshots=(
                result["q"].shape[0]
            ),
            rk4_steps=0,
        )
    ]


# ================================================================
# JAX
# ================================================================

def wait_for_jax(x):

    import jax

    return jax.tree_util.tree_map(
        lambda value:
            value.block_until_ready()
            if hasattr(
                value,
                "block_until_ready",
            )
            else value,
        x,
    )


def benchmark_models(args):

    import h5py
    import jax.numpy as jnp

    import experiments.plot_final_cfo_vs_geometry_ufno as suite

    resolution = int(
        args.resolution
    )

    dx = 5.0 / resolution
    dy = 5.0 / resolution

    # ------------------------------------------------------------
    # Configure existing suite for 64x64
    # ------------------------------------------------------------

    suite.RESOLUTION = resolution

    suite.X_MIN = -2.5
    suite.X_MAX = 2.5

    suite.Y_MIN = -2.5
    suite.Y_MAX = 2.5

    suite.DX = dx
    suite.DY = dy

    suite.CELL_AREA = (
        dx
        *
        dy
    )

    suite.GRAVITY = 1.0

    model_args = SimpleNamespace(
        cfo_ckpt_dir=str(
            resolve(
                args.cfo_ckpt_dir
            )
        ),
        cfo_prefix=(
            args.cfo_prefix
        ),
        final_ckpt=str(
            resolve(
                args.final_ckpt
            )
        ),
        lambda_pde=0.01,
        lambda_bed=0.70,
        lambda_wb=0.10,
        wb_eta0=1.5,
    )

    methods, states = (
        suite.build_models(
            model_args
        )
    )

    # ------------------------------------------------------------
    # Load hill_center
    # ------------------------------------------------------------

    data_path = resolve(
        args.counterfactual_data
    )

    with h5py.File(
        data_path,
        "r",
    ) as h5:

        q = np.asarray(
            h5[
                "cases/hill_center/q"
            ],
            dtype=np.float32,
        )

        b = np.asarray(
            h5[
                "cases/hill_center/bathymetry"
            ],
            dtype=np.float32,
        )

    if b.ndim == 2:

        b = b[
            ...,
            None
        ]

    q0 = jnp.asarray(
        q[
            0:
            1
        ],
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        b[
            None,
            ...
        ],
        dtype=jnp.float32,
    )

    n_times = int(
        q.shape[0]
    )

    rows = []

    configurations = [
        (
            "Bathy-CFO",
            methods["cfo"],
            states["cfo"],
        ),
        (
            "Final Geometry-U-FNO",
            methods["final"],
            states["final"],
        ),
    ]

    for (
        name,
        method,
        state,
    ) in configurations:

        print()
        print(
            "Warm-up:",
            name,
        )

        prediction = (
            method.uniform_inference(
                state,
                q0,
                trajectory_points_num=(
                    n_times
                ),
                steps_per_segment=(
                    args.steps_per_segment
                ),
                condition=(
                    condition
                ),
                method="RK4",
            )
        )

        wait_for_jax(
            prediction
        )

        values = []

        for repeat in range(
            args.repeats
        ):

            start = time.perf_counter()

            prediction = (
                method.uniform_inference(
                    state,
                    q0,
                    trajectory_points_num=(
                        n_times
                    ),
                    steps_per_segment=(
                        args.steps_per_segment
                    ),
                    condition=(
                        condition
                    ),
                    method="RK4",
                )
            )

            wait_for_jax(
                prediction
            )

            elapsed = (
                time.perf_counter()
                -
                start
            )

            values.append(
                elapsed
            )

            print(
                f"{name} run {repeat + 1}: "
                f"{elapsed:.6f} s"
            )

        rows.append(
            timing_row(
                name,
                values,
                resolution=resolution,
                snapshots=n_times,
                rk4_steps=(
                    args.steps_per_segment
                ),
            )
        )

    return rows


# ================================================================
# PLOT
# ================================================================

def plot_results(
    args,
    csv_path,
    output_dir,
):

    import matplotlib.pyplot as plt

    rows = load_existing(
        csv_path
    )

    lookup = {
        row["model"]:
            row
        for row in rows
    }

    required = [
        "PyClaw",
        "Bathy-CFO",
        "Final Geometry-U-FNO",
    ]

    missing = [
        name
        for name in required
        if name not in lookup
    ]

    if missing:

        raise RuntimeError(
            "Missing runtime results: "
            +
            ", ".join(
                missing
            )
        )

    names = required

    times = np.asarray(
        [
            float(
                lookup[name][
                    "mean_seconds"
                ]
            )
            for name in names
        ]
    )

    pyclaw_time = (
        times[0]
    )

    speedups = (
        pyclaw_time
        /
        times
    )

    # ------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.bar(
        names,
        times,
    )

    ax.set_ylabel(
        "Mean runtime (s)"
    )

    ax.set_title(
        "64x64 rollout runtime: t=0 to 1"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    runtime_path = (
        output_dir
        /
        "01_runtime_seconds.png"
    )

    fig.savefig(
        runtime_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ------------------------------------------------------------
    # Speedup
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )

    ax.bar(
        names,
        speedups,
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_ylabel(
        "Speedup relative to PyClaw"
    )

    ax.set_title(
        "64x64 computational speedup"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    speedup_path = (
        output_dir
        /
        "02_speedup_vs_pyclaw.png"
    )

    fig.savefig(
        speedup_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        "=" * 72
    )

    print(
        "RUNTIME SUMMARY"
    )

    print(
        "=" * 72
    )

    for (
        name,
        runtime,
        speedup,
    ) in zip(
        names,
        times,
        speedups,
    ):

        print(
            f"{name:24s} "
            f"{runtime:10.6f} s   "
            f"speedup={speedup:10.3f}x"
        )

    print()
    print(
        "Saved:",
        runtime_path
    )

    print(
        "Saved:",
        speedup_path
    )


def main():

    args = parse_args()

    output_dir, csv_path = (
        output_paths(
            args
        )
    )

    if args.backend == "pyclaw":

        rows = benchmark_pyclaw(
            args
        )

        save_rows(
            csv_path,
            rows,
        )

    elif args.backend == "models":

        rows = benchmark_models(
            args
        )

        save_rows(
            csv_path,
            rows,
        )

    else:

        plot_results(
            args,
            csv_path,
            output_dir,
        )

    print(
        "\nCSV:",
        csv_path,
    )


if __name__ == "__main__":

    main()