"""
Lake-at-rest benchmark.

Question
--------
If eta = h + b is constant and hu = hv = 0,
does the learned model keep the water stationary?

Exact solution
--------------
q(t) = q(0)

No PyClaw rollout is required because this equilibrium
has a known exact solution.

Models compared
---------------
1. Original FNO Bed-PI-CFO
2. FNO WB-Bed-PI-CFO
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# ROOT
# ================================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT),
    )


# ================================================================
# IMPORTS
# ================================================================

from models.fno import FNO2d

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bed-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_bed_lam07/"
            "seed0/best"
        ),
    )

    parser.add_argument(
        "--wb-ckpt-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lambda-wb",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--eta0",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--points",
        type=int,
        default=51,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "lake_at_rest"
        ),
    )

    return parser.parse_args()


# ================================================================
# TERRAIN FUNCTIONS
# ================================================================

def gaussian(
    X,
    Y,
    *,
    amplitude,
    xc,
    yc,
    sx,
    sy=None,
    angle_degrees=0.0,
):

    if sy is None:
        sy = sx

    angle = np.deg2rad(
        angle_degrees
    )

    dx = (
        X
        -
        xc
    )

    dy = (
        Y
        -
        yc
    )

    xr = (
        np.cos(
            angle
        )
        *
        dx
        +
        np.sin(
            angle
        )
        *
        dy
    )

    yr = (
        -np.sin(
            angle
        )
        *
        dx
        +
        np.cos(
            angle
        )
        *
        dy
    )

    exponent = (
        -0.5
        *
        (
            (
                xr
                /
                sx
            )**2
            +
            (
                yr
                /
                sy
            )**2
        )
    )

    return (
        amplitude
        *
        np.exp(
            exponent
        )
    )


def make_terrains(
    n=32,
):

    x = np.linspace(
        -2.5,
        2.5,
        n,
        dtype=np.float32,
    )

    y = np.linspace(
        -2.5,
        2.5,
        n,
        dtype=np.float32,
    )

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    terrains = {}

    terrains[
        "single_gaussian"
    ] = gaussian(
        X,
        Y,
        amplitude=0.20,
        xc=0.0,
        yc=0.0,
        sx=0.50,
    )

    terrains[
        "two_hills"
    ] = (
        gaussian(
            X,
            Y,
            amplitude=0.15,
            xc=-0.8,
            yc=-0.2,
            sx=0.42,
        )
        +
        gaussian(
            X,
            Y,
            amplitude=0.13,
            xc=0.8,
            yc=0.3,
            sx=0.50,
        )
    )

    terrains[
        "narrow_tall"
    ] = gaussian(
        X,
        Y,
        amplitude=0.34,
        xc=-0.55,
        yc=0.20,
        sx=0.24,
    )

    terrains[
        "elongated_ridge"
    ] = gaussian(
        X,
        Y,
        amplitude=0.22,
        xc=-0.4,
        yc=0.0,
        sx=0.22,
        sy=0.95,
    )

    terrains[
        "rotated_ridge"
    ] = gaussian(
        X,
        Y,
        amplitude=0.22,
        xc=0.0,
        yc=0.0,
        sx=0.22,
        sy=0.90,
        angle_degrees=45.0,
    )

    terrains[
        "multi_hill"
    ] = (
        gaussian(
            X,
            Y,
            amplitude=0.12,
            xc=-0.85,
            yc=-0.40,
            sx=0.35,
        )
        +
        gaussian(
            X,
            Y,
            amplitude=0.15,
            xc=0.0,
            yc=0.65,
            sx=0.45,
        )
        +
        gaussian(
            X,
            Y,
            amplitude=0.10,
            xc=0.80,
            yc=-0.25,
            sx=0.32,
        )
    )

    terrains = {
        key:
        np.asarray(
            value,
            dtype=np.float32,
        )
        for key, value
        in terrains.items()
    }

    return (
        x,
        y,
        terrains,
    )


# ================================================================
# EQUILIBRIUM STATE
# ================================================================

def make_equilibrium(
    bathymetry,
    eta0,
):

    h = (
        eta0
        -
        bathymetry
    )

    if np.min(
        h
    ) <= 0:

        raise ValueError(
            "eta0 is too small for "
            "this bathymetry."
        )

    zero = np.zeros_like(
        h
    )

    return np.stack(
        [
            h,
            zero,
            zero,
        ],
        axis=-1,
    ).astype(
        np.float32
    )


# ================================================================
# MODEL BUILDERS
# ================================================================

def build_model():

    return FNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        use_condition=True,
        use_time=True,
    )


def build_bed_method():

    model = build_model()

    return (
        BathymetryBedRegularizedPICFO(
            model=model,
            input_shape=(
                32,
                32,
                3,
            ),
            condition_shape=(
                32,
                32,
                1,
            ),
            gamma=1e-5,
            spline_type="quintic",
            lambda_pde=0.03,
            lambda_bed=0.70,
            dx=0.15625,
            dy=0.15625,
            gravity=1.0,
        )
    )


def build_wb_method(
    lambda_wb,
    eta0,
):

    model = build_model()

    return (
        WellBalancedBathymetryBedPICFO(
            model=model,
            input_shape=(
                32,
                32,
                3,
            ),
            condition_shape=(
                32,
                32,
                1,
            ),
            gamma=1e-5,
            spline_type="quintic",
            lambda_pde=0.03,
            lambda_bed=0.70,
            lambda_wb=lambda_wb,
            wb_eta0=eta0,
            dx=0.15625,
            dy=0.15625,
            gravity=1.0,
        )
    )


# ================================================================
# RESTORE CHECKPOINT MANAGER DIRECTORY
# ================================================================

def restore_checkpoint(
    method,
    manager_directory,
):

    manager_directory = Path(
        manager_directory
    )

    state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        state,
        ckpt_dir=str(
            manager_directory.parent
        ),
        prefix=(
            manager_directory.name
        ),
        step=None,
        max_to_keep=1,
    )


# ================================================================
# METRICS
# ================================================================

def equilibrium_metrics(
    prediction,
    bathymetry,
    q0,
):

    truth = np.repeat(
        q0[
            None,
            ...
        ],
        prediction.shape[
            0
        ],
        axis=0,
    )

    hu = prediction[
        ...,
        1
    ]

    hv = prediction[
        ...,
        2
    ]

    eta = (
        prediction[
            ...,
            0
        ]
        +
        bathymetry[
            None,
            ...
        ]
    )

    eta_initial = (
        q0[
            ...,
            0
        ]
        +
        bathymetry
    )

    eta_drift = np.abs(
        eta
        -
        eta_initial[
            None,
            ...
        ]
    )

    relative_l2 = (
        np.linalg.norm(
            prediction
            -
            truth
        )
        /
        max(
            np.linalg.norm(
                truth
            ),
            1e-12,
        )
    )

    momentum_speed = np.sqrt(
        hu**2
        +
        hv**2
    )

    return {
        "relative_l2":
            float(
                relative_l2
            ),

        "max_abs_hu":
            float(
                np.max(
                    np.abs(
                        hu
                    )
                )
            ),

        "max_abs_hv":
            float(
                np.max(
                    np.abs(
                        hv
                    )
                )
            ),

        "max_spurious_momentum":
            float(
                np.max(
                    momentum_speed
                )
            ),

        "max_eta_drift":
            float(
                np.max(
                    eta_drift
                )
            ),

        "mean_eta_drift":
            float(
                np.mean(
                    eta_drift
                )
            ),
    }


# ================================================================
# ROLLOUT
# ================================================================

def run_model(
    method,
    state,
    q0,
    bathymetry,
    *,
    points,
    steps_per_segment,
):

    q0_batch = jnp.asarray(
        q0[
            None,
            ...
        ],
        dtype=jnp.float32,
    )

    b_batch = jnp.asarray(
        bathymetry[
            None,
            ...,
            None,
        ],
        dtype=jnp.float32,
    )

    pred = method.uniform_inference(
        state,
        q0_batch,
        trajectory_points_num=points,
        steps_per_segment=(
            steps_per_segment
        ),
        condition=b_batch,
        method="RK4",
    )

    return np.asarray(
        pred[
            0
        ]
    )


# ================================================================
# SAVE CSV
# ================================================================

def save_csv(
    rows,
    path,
):

    columns = [
        "terrain",
        "model",
        "relative_l2",
        "max_abs_hu",
        "max_abs_hv",
        "max_spurious_momentum",
        "max_eta_drift",
        "mean_eta_drift",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ================================================================
# PLOTS
# ================================================================

def make_plots(
    rows,
    output_dir,
):

    terrains = []

    for row in rows:

        if (
            row[
                "terrain"
            ]
            not in terrains
        ):
            terrains.append(
                row[
                    "terrain"
                ]
            )

    models = [
        "Bed-PI-CFO",
        "WB-Bed-PI-CFO",
    ]

    # ------------------------------------------------------------
    # ETA DRIFT
    # ------------------------------------------------------------

    x = np.arange(
        len(
            terrains
        )
    )

    width = 0.35

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    for model_index, model in enumerate(
        models
    ):

        values = []

        for terrain in terrains:

            match = [
                row
                for row
                in rows
                if (
                    row[
                        "terrain"
                    ]
                    ==
                    terrain
                    and
                    row[
                        "model"
                    ]
                    ==
                    model
                )
            ][0]

            values.append(
                float(
                    match[
                        "max_eta_drift"
                    ]
                )
            )

        plt.bar(
            x
            +
            (
                model_index
                -
                0.5
            )
            *
            width,
            values,
            width=width,
            label=model,
        )

    plt.yscale(
        "log"
    )

    plt.xticks(
        x,
        terrains,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Maximum |eta(t) - eta(0)|"
    )

    plt.title(
        "Lake-at-rest free-surface drift"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "01_lake_at_rest_eta_drift.png",
        dpi=220,
    )

    plt.close()

    # ------------------------------------------------------------
    # SPURIOUS MOMENTUM
    # ------------------------------------------------------------

    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    for model_index, model in enumerate(
        models
    ):

        values = []

        for terrain in terrains:

            match = [
                row
                for row
                in rows
                if (
                    row[
                        "terrain"
                    ]
                    ==
                    terrain
                    and
                    row[
                        "model"
                    ]
                    ==
                    model
                )
            ][0]

            values.append(
                float(
                    match[
                        "max_spurious_momentum"
                    ]
                )
            )

        plt.bar(
            x
            +
            (
                model_index
                -
                0.5
            )
            *
            width,
            values,
            width=width,
            label=model,
        )

    plt.yscale(
        "log"
    )

    plt.xticks(
        x,
        terrains,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Maximum spurious momentum"
    )

    plt.title(
        "Lake-at-rest spurious motion"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir
        /
        "02_lake_at_rest_spurious_momentum.png",
        dpi=220,
    )

    plt.close()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    output_dir = (
        ROOT
        /
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bed_method = (
        build_bed_method()
    )

    wb_method = (
        build_wb_method(
            args.lambda_wb,
            args.eta0,
        )
    )

    print(
        "Restoring original Bed-PI-CFO..."
    )

    bed_state = (
        restore_checkpoint(
            bed_method,
            ROOT
            /
            args.bed_ckpt_dir,
        )
    )

    print(
        "Restoring WB-Bed-PI-CFO..."
    )

    wb_state = (
        restore_checkpoint(
            wb_method,
            ROOT
            /
            args.wb_ckpt_dir,
        )
    )

    _, _, terrains = (
        make_terrains(
            n=32
        )
    )

    rows = []

    for name, bathymetry in (
        terrains.items()
    ):

        print()
        print(
            "Terrain:",
            name,
        )

        q0 = make_equilibrium(
            bathymetry,
            args.eta0,
        )

        bed_prediction = (
            run_model(
                bed_method,
                bed_state,
                q0,
                bathymetry,
                points=args.points,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )
        )

        wb_prediction = (
            run_model(
                wb_method,
                wb_state,
                q0,
                bathymetry,
                points=args.points,
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )
        )

        bed_metrics = (
            equilibrium_metrics(
                bed_prediction,
                bathymetry,
                q0,
            )
        )

        wb_metrics = (
            equilibrium_metrics(
                wb_prediction,
                bathymetry,
                q0,
            )
        )

        bed_row = {
            "terrain":
                name,

            "model":
                "Bed-PI-CFO",

            **bed_metrics,
        }

        wb_row = {
            "terrain":
                name,

            "model":
                "WB-Bed-PI-CFO",

            **wb_metrics,
        }

        rows.append(
            bed_row
        )

        rows.append(
            wb_row
        )

        print(
            "Bed max eta drift:",
            f"{bed_metrics['max_eta_drift']:.6e}",
        )

        print(
            "WB  max eta drift:",
            f"{wb_metrics['max_eta_drift']:.6e}",
        )

        print(
            "Bed spurious momentum:",
            f"{bed_metrics['max_spurious_momentum']:.6e}",
        )

        print(
            "WB  spurious momentum:",
            f"{wb_metrics['max_spurious_momentum']:.6e}",
        )

    save_csv(
        rows,
        output_dir
        /
        "lake_at_rest_metrics.csv",
    )

    make_plots(
        rows,
        output_dir,
    )

    print()
    print(
        "=================================="
    )

    print(
        "LAKE-AT-REST BENCHMARK COMPLETE"
    )

    print(
        "=================================="
    )

    print(
        "Results:",
        output_dir,
    )


if __name__ == "__main__":
    main()