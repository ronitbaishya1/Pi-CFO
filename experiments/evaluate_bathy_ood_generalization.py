"""
Evaluate frozen PI-CFO and Bed-PI-CFO on unseen bathymetry.

NO TRAINING OCCURS IN THIS FILE.

Goal
----
Answer:

    Did Bed-PI-CFO learn a general terrain-conditioned operator,
    or only the single-Gaussian training family?

Evaluation families
-------------------
ID control:
    id_gaussian

OOD:
    two_hills
    narrow_tall
    elongated_ridge
    rotated_ridge
    multi_hill

Metrics
-------
1. Full rollout Relative L2
2. Isolated terrain-effect error:
       delta q = q_terrain - q_flat
3. Terrain-induced velocity cosine similarity
4. Terrain-induced velocity magnitude ratio
5. Final mass drift
6. Same-ground-truth SWE residual
7. Direct bed-response MSE
8. Direct bed-response cosine
9. Direct bed-response magnitude ratio

Output
------
results/bathy_ood_generalization/

    per_case_metrics.csv
    family_summary.csv

    01_terrain_gallery.png
    02_rollout_l2_by_family.png
    03_terrain_effect_error_by_family.png
    04_velocity_cosine_by_family.png
    05_velocity_magnitude_by_family.png
    06_direct_bed_cosine_by_family.png
    07_direct_bed_magnitude_by_family.png
    08_representative_terrain_effects.png
"""

from __future__ import annotations

import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path
import sys

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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

from bathy_pi_cfo import (
    BathymetryPhysicsInformedCFO,
)

from bathy_bed_pi_cfo import (
    BathymetryBedRegularizedPICFO,
)

from models.factory import (
    build_model,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)

from utils.physics_swe_bathy import (
    swe_bathy_residual,
    well_balanced_bed_source,
)


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_32.h5"
        ),
    )

    parser.add_argument(
        "--pi-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_pi_lam003/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--bed-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_bed_lam07/"
            "seed0/"
            "best"
        ),
    )

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
        "--time",
        type=float,
        default=0.50,
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

    parser.add_argument(
        "--physics-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "bathy_ood_generalization"
        ),
    )

    return parser.parse_args()


# ================================================================
# PATH
# ================================================================

def resolve_path(
    value,
):

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
# LOAD DATASET
# ================================================================

def load_dataset(
    path,
):

    result = {
        "cases":
            {}
    }

    with h5py.File(
        path,
        "r",
    ) as h5:

        result[
            "x"
        ] = np.asarray(
            h5[
                "x"
            ],
            dtype=np.float32,
        )

        result[
            "y"
        ] = np.asarray(
            h5[
                "y"
            ],
            dtype=np.float32,
        )

        result[
            "time"
        ] = np.asarray(
            h5[
                "time"
            ],
            dtype=np.float32,
        )

        for case_name in (
            h5[
                "cases"
            ].keys()
        ):

            group = (
                h5[
                    "cases"
                ][
                    case_name
                ]
            )

            family = group.attrs[
                "family"
            ]

            if isinstance(
                family,
                bytes,
            ):

                family = (
                    family.decode()
                )

            result[
                "cases"
            ][
                case_name
            ] = {
                "q":
                    np.asarray(
                        group[
                            "q"
                        ],
                        dtype=np.float32,
                    ),

                "bathymetry":
                    np.asarray(
                        group[
                            "bathymetry"
                        ],
                        dtype=np.float32,
                    ),

                "family":
                    str(
                        family
                    ),
            }

    return result


# ================================================================
# BASIC METRICS
# ================================================================

def relative_error(
    truth,
    prediction,
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
            1e-12,
        )
    )


def vector_cosine(
    true_field,
    predicted_field,
):

    true_vector = (
        true_field.ravel()
    )

    pred_vector = (
        predicted_field.ravel()
    )

    denominator = (
        np.linalg.norm(
            true_vector
        )
        *
        np.linalg.norm(
            pred_vector
        )
    )

    return float(
        np.dot(
            true_vector,
            pred_vector,
        )
        /
        max(
            denominator,
            1e-12,
        )
    )


def norm_ratio(
    truth,
    prediction,
):

    return float(
        np.linalg.norm(
            prediction.ravel()
        )
        /
        max(
            np.linalg.norm(
                truth.ravel()
            ),
            1e-12,
        )
    )


# ================================================================
# VELOCITY
# ================================================================

def velocity(
    q,
):

    h = np.maximum(
        q[
            ...,
            0
        ],
        1e-6,
    )

    u = (
        q[
            ...,
            1
        ]
        /
        h
    )

    v = (
        q[
            ...,
            2
        ]
        /
        h
    )

    speed = np.sqrt(
        u**2
        +
        v**2
    )

    return (
        u,
        v,
        speed,
    )


# ================================================================
# MASS
# ================================================================

def final_mass_drift(
    q,
    dx,
    dy,
):

    h = q[
        ...,
        0
    ]

    mass = (
        np.sum(
            h,
            axis=(
                1,
                2,
            ),
        )
        *
        dx
        *
        dy
    )

    return float(
        np.abs(
            mass[
                -1
            ]
            -
            mass[
                0
            ]
        )
        /
        max(
            np.abs(
                mass[
                    0
                ]
            ),
            1e-12,
        )
    )


# ================================================================
# BUILD MODELS
# ================================================================

def build_pi(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return BathymetryPhysicsInformedCFO(
        model=model,
        input_shape=input_shape,
        condition_shape=(
            condition_shape
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=(
            args.lambda_pde
        ),
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )


def build_bed_pi(
    input_shape,
    condition_shape,
    args,
):

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return BathymetryBedRegularizedPICFO(
        model=model,
        input_shape=input_shape,
        condition_shape=(
            condition_shape
        ),
        gamma=1e-5,
        spline_type="quintic",
        lambda_pde=(
            args.lambda_pde
        ),
        lambda_bed=(
            args.lambda_bed
        ),
        dx=args.dx,
        dy=args.dy,
        gravity=args.gravity,
    )


# ================================================================
# RESTORE
# ================================================================

def restore_state(
    method,
    directory,
    prefix,
):

    target = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target,
        str(
            directory
        ),
        prefix=prefix,
        step=None,
        max_to_keep=1,
    )


# ================================================================
# MODEL ROLLOUT
# ================================================================

def rollout(
    method,
    state,
    case,
):

    q_true = case[
        "q"
    ]

    bathymetry = case[
        "bathymetry"
    ]

    prediction = method.uniform_inference(
        state,
        q_true[
            0
        ][
            None,
            ...
        ],
        trajectory_points_num=(
            q_true.shape[
                0
            ]
        ),
        steps_per_segment=2,
        condition=bathymetry[
            None,
            ...
        ],
        method="RK4",
    )

    return np.asarray(
        prediction[
            0
        ],
        dtype=np.float32,
    )


# ================================================================
# SAME-GROUND-TRUTH PHYSICS + DIRECT BED METRICS
# ================================================================

def physics_metrics(
    method,
    state,
    q_true,
    bathymetry,
    time,
    args,
):

    T = q_true.shape[
        0
    ]

    pde_squared = 0.0
    pde_count = 0

    bed_squared = 0.0
    bed_count = 0

    bed_dot = 0.0
    bed_true_norm_sq = 0.0
    bed_pred_norm_sq = 0.0

    for start in range(
        0,
        T,
        args.physics_batch_size,
    ):

        end = min(
            start
            +
            args.physics_batch_size,
            T,
        )

        q = jnp.asarray(
            q_true[
                start:end
            ]
        )

        t = jnp.asarray(
            time[
                start:end
            ]
        )

        b = jnp.repeat(
            jnp.asarray(
                bathymetry[
                    None,
                    ...
                ]
            ),
            repeats=(
                end
                -
                start
            ),
            axis=0,
        )

        # --------------------------------------------------------
        # Model vector field with actual terrain
        # --------------------------------------------------------

        q_t_b = method._model_apply(
            state.params,
            q,
            t,
            b,
        )

        # --------------------------------------------------------
        # Same GT PDE residual
        # --------------------------------------------------------

        residual = swe_bathy_residual(
            q,
            q_t_b,
            b,
            dx=args.dx,
            dy=args.dy,
            g=args.gravity,
        )

        residual_np = np.asarray(
            residual,
            dtype=np.float64,
        )

        pde_squared += float(
            np.sum(
                residual_np**2
            )
        )

        pde_count += int(
            residual_np.size
        )

        # --------------------------------------------------------
        # Same q and t, remove terrain
        # --------------------------------------------------------

        q_t_zero = method._model_apply(
            state.params,
            q,
            t,
            jnp.zeros_like(
                b
            ),
        )

        delta_q_t = (
            q_t_b
            -
            q_t_zero
        )

        # --------------------------------------------------------
        # Expected SWE bed response
        # --------------------------------------------------------

        (
            source_x,
            source_y,
        ) = well_balanced_bed_source(
            q,
            b,
            dx=args.dx,
            dy=args.dy,
            g=args.gravity,
        )

        expected = jnp.stack(
            [
                -source_x,
                -source_y,
            ],
            axis=-1,
        )

        predicted = delta_q_t[
            :,
            1:-1,
            1:-1,
            1:3,
        ]

        expected_np = np.asarray(
            expected,
            dtype=np.float64,
        )

        predicted_np = np.asarray(
            predicted,
            dtype=np.float64,
        )

        difference = (
            predicted_np
            -
            expected_np
        )

        bed_squared += float(
            np.sum(
                difference**2
            )
        )

        bed_count += int(
            difference.size
        )

        bed_dot += float(
            np.sum(
                expected_np
                *
                predicted_np
            )
        )

        bed_true_norm_sq += float(
            np.sum(
                expected_np**2
            )
        )

        bed_pred_norm_sq += float(
            np.sum(
                predicted_np**2
            )
        )

    pde_mse = (
        pde_squared
        /
        max(
            pde_count,
            1,
        )
    )

    bed_mse = (
        bed_squared
        /
        max(
            bed_count,
            1,
        )
    )

    bed_cosine = (
        bed_dot
        /
        max(
            np.sqrt(
                bed_true_norm_sq
                *
                bed_pred_norm_sq
            ),
            1e-12,
        )
    )

    bed_magnitude = (
        np.sqrt(
            bed_pred_norm_sq
        )
        /
        max(
            np.sqrt(
                bed_true_norm_sq
            ),
            1e-12,
        )
    )

    return {
        "same_gt_pde_mse":
            float(
                pde_mse
            ),

        "direct_bed_mse":
            float(
                bed_mse
            ),

        "direct_bed_cosine":
            float(
                bed_cosine
            ),

        "direct_bed_magnitude_ratio":
            float(
                bed_magnitude
            ),
    }


# ================================================================
# ONE CASE METRICS
# ================================================================

def case_metrics(
    *,
    case_name,
    case,
    flat_case,
    prediction,
    flat_prediction,
    method,
    state,
    time,
    time_index,
    args,
):

    true_q = case[
        "q"
    ]

    true_flat = flat_case[
        "q"
    ]

    # ============================================================
    # COMPLETE ROLLOUT
    # ============================================================

    rollout_rel_l2 = relative_error(
        true_q,
        prediction,
    )

    # ============================================================
    # ISOLATED TERRAIN EFFECT
    # ============================================================

    true_delta_q = (
        true_q
        -
        true_flat
    )

    pred_delta_q = (
        prediction
        -
        flat_prediction
    )

    terrain_delta_error = relative_error(
        true_delta_q,
        pred_delta_q,
    )

    # ============================================================
    # TERRAIN-INDUCED VELOCITY AT SELECTED TIME
    # ============================================================

    true_flat_velocity = velocity(
        true_flat[
            time_index
        ]
    )

    true_case_velocity = velocity(
        true_q[
            time_index
        ]
    )

    pred_flat_velocity = velocity(
        flat_prediction[
            time_index
        ]
    )

    pred_case_velocity = velocity(
        prediction[
            time_index
        ]
    )

    true_velocity_effect = np.stack(
        [
            true_case_velocity[
                0
            ]
            -
            true_flat_velocity[
                0
            ],

            true_case_velocity[
                1
            ]
            -
            true_flat_velocity[
                1
            ],
        ],
        axis=-1,
    )

    pred_velocity_effect = np.stack(
        [
            pred_case_velocity[
                0
            ]
            -
            pred_flat_velocity[
                0
            ],

            pred_case_velocity[
                1
            ]
            -
            pred_flat_velocity[
                1
            ],
        ],
        axis=-1,
    )

    velocity_cosine = vector_cosine(
        true_velocity_effect,
        pred_velocity_effect,
    )

    velocity_magnitude = norm_ratio(
        true_velocity_effect,
        pred_velocity_effect,
    )

    # ============================================================
    # PHYSICS
    # ============================================================

    physics = physics_metrics(
        method,
        state,
        true_q,
        case[
            "bathymetry"
        ],
        time,
        args,
    )

    return {
        "rollout_rel_l2":
            rollout_rel_l2,

        "terrain_delta_q_error":
            terrain_delta_error,

        "velocity_cosine":
            velocity_cosine,

        "velocity_magnitude_ratio":
            velocity_magnitude,

        "final_mass_drift":
            final_mass_drift(
                prediction,
                args.dx,
                args.dy,
            ),

        **physics,
    }


# ================================================================
# CSV
# ================================================================

def save_csv(
    rows,
    path,
):

    with path.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ================================================================
# FAMILY SUMMARY
# ================================================================

def family_summary(
    rows,
):

    metric_names = [
        "rollout_rel_l2",
        "terrain_delta_q_error",
        "velocity_cosine",
        "velocity_magnitude_ratio",
        "final_mass_drift",
        "same_gt_pde_mse",
        "direct_bed_mse",
        "direct_bed_cosine",
        "direct_bed_magnitude_ratio",
    ]

    grouped = defaultdict(
        list
    )

    for row in rows:

        key = (
            row[
                "family"
            ],
            row[
                "model"
            ],
        )

        grouped[
            key
        ].append(
            row
        )

    output = []

    for (
        family,
        model,
    ), family_rows in sorted(
        grouped.items()
    ):

        summary = {
            "family":
                family,

            "model":
                model,

            "n_cases":
                len(
                    family_rows
                ),
        }

        for metric in metric_names:

            values = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row
                    in family_rows
                ],
                dtype=np.float64,
            )

            summary[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            summary[
                f"{metric}_std"
            ] = float(
                np.std(
                    values
                )
            )

        output.append(
            summary
        )

    return output


# ================================================================
# FAMILY ORDER
# ================================================================

FAMILY_ORDER = [
    "id_gaussian",
    "two_hills",
    "narrow_tall",
    "elongated_ridge",
    "rotated_ridge",
    "multi_hill",
]


# ================================================================
# SUMMARY METRIC PLOT
# ================================================================

def plot_family_metric(
    summary_rows,
    *,
    metric,
    ylabel,
    title,
    output_path,
    ideal_line=None,
):

    models = [
        "PI-CFO",
        "Bed-PI-CFO",
    ]

    x = np.arange(
        len(
            FAMILY_ORDER
        )
    )

    width = 0.36

    fig = plt.figure(
        figsize=(
            11,
            5,
        )
    )

    for model_index, model in enumerate(
        models
    ):

        means = []
        errors = []

        for family in FAMILY_ORDER:

            match = [
                row
                for row
                in summary_rows
                if row[
                    "family"
                ]
                ==
                family
                and
                row[
                    "model"
                ]
                ==
                model
            ]

            if not match:

                means.append(
                    np.nan
                )

                errors.append(
                    np.nan
                )

            else:

                row = match[
                    0
                ]

                means.append(
                    row[
                        f"{metric}_mean"
                    ]
                )

                errors.append(
                    row[
                        f"{metric}_std"
                    ]
                )

        offset = (
            model_index
            -
            0.5
        ) * width

        plt.bar(
            x
            +
            offset,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=model,
        )

    if ideal_line is not None:

        plt.axhline(
            ideal_line,
            linestyle="--",
            label="Ideal",
        )

    plt.xticks(
        x,
        FAMILY_ORDER,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# TERRAIN GALLERY
# ================================================================

def plot_terrain_gallery(
    data,
    output_dir,
):

    representative = {}

    for family in FAMILY_ORDER:

        names = sorted(
            [
                name
                for name, case
                in data[
                    "cases"
                ].items()
                if case[
                    "family"
                ]
                ==
                family
            ]
        )

        if names:

            representative[
                family
            ] = names[
                0
            ]

    X, Y = np.meshgrid(
        data[
            "x"
        ],
        data[
            "y"
        ],
        indexing="ij",
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            13,
            8,
        ),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for (
        ax,
        family,
    ) in zip(
        axes,
        FAMILY_ORDER,
    ):

        case_name = representative[
            family
        ]

        bathymetry = (
            data[
                "cases"
            ][
                case_name
            ][
                "bathymetry"
            ][
                ...,
                0
            ]
        )

        mesh = ax.pcolormesh(
            X,
            Y,
            bathymetry,
            shading="auto",
        )

        fig.colorbar(
            mesh,
            ax=ax,
            label="b",
        )

        ax.set_title(
            family
        )

        ax.set_xlabel(
            "x"
        )

        ax.set_ylabel(
            "y"
        )

        ax.set_aspect(
            "equal"
        )

    fig.suptitle(
        "Bathymetry generalization benchmark"
    )

    fig.savefig(
        output_dir
        /
        "01_terrain_gallery.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# REPRESENTATIVE TERRAIN EFFECT MAP
# ================================================================

def plot_representative_effects(
    data,
    pi_predictions,
    bed_predictions,
    time_index,
    output_dir,
):

    X, Y = np.meshgrid(
        data[
            "x"
        ],
        data[
            "y"
        ],
        indexing="ij",
    )

    flat_true = data[
        "cases"
    ][
        "flat"
    ][
        "q"
    ]

    flat_pi = pi_predictions[
        "flat"
    ]

    flat_bed = bed_predictions[
        "flat"
    ]

    flat_true_speed = velocity(
        flat_true[
            time_index
        ]
    )[
        2
    ]

    flat_pi_speed = velocity(
        flat_pi[
            time_index
        ]
    )[
        2
    ]

    flat_bed_speed = velocity(
        flat_bed[
            time_index
        ]
    )[
        2
    ]

    families = [
        "two_hills",
        "narrow_tall",
        "elongated_ridge",
        "rotated_ridge",
        "multi_hill",
    ]

    fig, axes = plt.subplots(
        len(
            families
        ),
        3,
        figsize=(
            14,
            20,
        ),
        constrained_layout=True,
    )

    column_names = [
        "PyClaw",
        "PI-CFO",
        "Bed-PI-CFO",
    ]

    for row, family in enumerate(
        families
    ):

        case_name = sorted(
            [
                name
                for name, case
                in data[
                    "cases"
                ].items()
                if case[
                    "family"
                ]
                ==
                family
            ]
        )[
            0
        ]

        case = data[
            "cases"
        ][
            case_name
        ]

        true_speed = velocity(
            case[
                "q"
            ][
                time_index
            ]
        )[
            2
        ]

        pi_speed = velocity(
            pi_predictions[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        bed_speed = velocity(
            bed_predictions[
                case_name
            ][
                time_index
            ]
        )[
            2
        ]

        effects = [
            true_speed
            -
            flat_true_speed,

            pi_speed
            -
            flat_pi_speed,

            bed_speed
            -
            flat_bed_speed,
        ]

        maximum = max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            )
            for field
            in effects
        )

        bathymetry = case[
            "bathymetry"
        ][
            ...,
            0
        ]

        bathy_max = float(
            np.max(
                bathymetry
            )
        )

        contour_levels = np.linspace(
            0.2
            *
            bathy_max,
            0.9
            *
            bathy_max,
            5,
        )

        for column in range(
            3
        ):

            ax = axes[
                row,
                column
            ]

            mesh = ax.pcolormesh(
                X,
                Y,
                effects[
                    column
                ],
                shading="auto",
                cmap="coolwarm",
                vmin=-maximum,
                vmax=maximum,
            )

            ax.contour(
                X,
                Y,
                bathymetry,
                levels=(
                    contour_levels
                ),
                linewidths=0.8,
            )

            fig.colorbar(
                mesh,
                ax=ax,
            )

            ax.set_title(
                f"{family}\n"
                f"{column_names[column]}"
            )

            ax.set_xlabel(
                "x"
            )

            ax.set_ylabel(
                "y"
            )

            ax.set_aspect(
                "equal"
            )

    fig.suptitle(
        "Terrain-induced speed on unseen terrain"
    )

    fig.savefig(
        output_dir
        /
        "08_representative_terrain_effects.png",
        dpi=220,
    )

    plt.close(
        fig
    )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    dataset_path = resolve_path(
        args.dataset
    )

    pi_checkpoint = resolve_path(
        args.pi_ckpt_dir
    )

    bed_checkpoint = resolve_path(
        args.bed_ckpt_dir
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # PATH CHECK
    # ============================================================

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{dataset_path}"
        )

    if not pi_checkpoint.exists():

        raise FileNotFoundError(
            f"PI checkpoint not found:\n"
            f"{pi_checkpoint}"
        )

    if not bed_checkpoint.exists():

        raise FileNotFoundError(
            f"Bed-PI checkpoint not found:\n"
            f"{bed_checkpoint}"
        )

    # ============================================================
    # LOAD DATA
    # ============================================================

    print(
        "\nLoading OOD dataset..."
    )

    data = load_dataset(
        dataset_path
    )

    cases = data[
        "cases"
    ]

    time = data[
        "time"
    ]

    if "flat" not in cases:

        raise KeyError(
            "OOD dataset must contain "
            "a case named 'flat'."
        )

    time_index = int(
        np.argmin(
            np.abs(
                time
                -
                args.time
            )
        )
    )

    print(
        f"Selected diagnostic time: "
        f"{time[time_index]:.3f}"
    )

    flat_case = cases[
        "flat"
    ]

    input_shape = tuple(
        flat_case[
            "q"
        ].shape[
            1:
        ]
    )

    condition_shape = tuple(
        flat_case[
            "bathymetry"
        ].shape
    )

    # ============================================================
    # RESTORE ORIGINAL PI
    # ============================================================

    print(
        "\nRestoring PI-CFO..."
    )

    pi_method = build_pi(
        input_shape,
        condition_shape,
        args,
    )

    pi_state = restore_state(
        pi_method,
        pi_checkpoint,
        "bathy_pi_cfo",
    )

    # ============================================================
    # RESTORE BED PI
    # ============================================================

    print(
        "Restoring Bed-PI-CFO..."
    )

    bed_method = build_bed_pi(
        input_shape,
        condition_shape,
        args,
    )

    bed_state = restore_state(
        bed_method,
        bed_checkpoint,
        "bathy_bed_pi",
    )

    # ============================================================
    # RUN FROZEN MODEL ON EVERY TERRAIN
    # ============================================================

    print(
        "\nRunning frozen model rollouts..."
    )

    pi_predictions = {}

    bed_predictions = {}

    for case_name in sorted(
        cases.keys()
    ):

        print(
            f"  {case_name}"
        )

        pi_predictions[
            case_name
        ] = rollout(
            pi_method,
            pi_state,
            cases[
                case_name
            ],
        )

        bed_predictions[
            case_name
        ] = rollout(
            bed_method,
            bed_state,
            cases[
                case_name
            ],
        )

    # ============================================================
    # FLAT PREDICTIONS
    # ============================================================

    flat_pi_prediction = (
        pi_predictions[
            "flat"
        ]
    )

    flat_bed_prediction = (
        bed_predictions[
            "flat"
        ]
    )

    # ============================================================
    # CASE METRICS
    # ============================================================

    rows = []

    print(
        "\nComputing metrics..."
    )

    for case_name in sorted(
        cases.keys()
    ):

        if case_name == "flat":

            continue

        case = cases[
            case_name
        ]

        print(
            f"  {case_name}"
        )

        # --------------------------------------------------------
        # PI-CFO
        # --------------------------------------------------------

        pi_metrics = case_metrics(
            case_name=case_name,
            case=case,
            flat_case=flat_case,
            prediction=(
                pi_predictions[
                    case_name
                ]
            ),
            flat_prediction=(
                flat_pi_prediction
            ),
            method=pi_method,
            state=pi_state,
            time=time,
            time_index=time_index,
            args=args,
        )

        rows.append(
            {
                "case":
                    case_name,

                "family":
                    case[
                        "family"
                    ],

                "model":
                    "PI-CFO",

                **pi_metrics,
            }
        )

        # --------------------------------------------------------
        # BED PI-CFO
        # --------------------------------------------------------

        bed_metrics = case_metrics(
            case_name=case_name,
            case=case,
            flat_case=flat_case,
            prediction=(
                bed_predictions[
                    case_name
                ]
            ),
            flat_prediction=(
                flat_bed_prediction
            ),
            method=bed_method,
            state=bed_state,
            time=time,
            time_index=time_index,
            args=args,
        )

        rows.append(
            {
                "case":
                    case_name,

                "family":
                    case[
                        "family"
                    ],

                "model":
                    "Bed-PI-CFO",

                **bed_metrics,
            }
        )

    # ============================================================
    # SAVE PER-CASE TABLE
    # ============================================================

    per_case_path = (
        output_dir
        /
        "per_case_metrics.csv"
    )

    save_csv(
        rows,
        per_case_path,
    )

    # ============================================================
    # FAMILY SUMMARY
    # ============================================================

    summary_rows = family_summary(
        rows
    )

    summary_path = (
        output_dir
        /
        "family_summary.csv"
    )

    save_csv(
        summary_rows,
        summary_path,
    )

    # ============================================================
    # PLOTS
    # ============================================================

    plot_terrain_gallery(
        data,
        output_dir,
    )

    plot_family_metric(
        summary_rows,
        metric="rollout_rel_l2",
        ylabel="Relative L2",
        title=(
            "Full rollout accuracy "
            "on unseen bathymetry"
        ),
        output_path=(
            output_dir
            /
            "02_rollout_l2_by_family.png"
        ),
    )

    plot_family_metric(
        summary_rows,
        metric="terrain_delta_q_error",
        ylabel=(
            "Relative error in "
            "terrain-induced delta q"
        ),
        title=(
            "Accuracy of isolated "
            "terrain effect"
        ),
        output_path=(
            output_dir
            /
            "03_terrain_effect_error_by_family.png"
        ),
    )

    plot_family_metric(
        summary_rows,
        metric="velocity_cosine",
        ylabel="Cosine similarity",
        title=(
            "Terrain-induced velocity "
            "direction agreement"
        ),
        output_path=(
            output_dir
            /
            "04_velocity_cosine_by_family.png"
        ),
        ideal_line=1.0,
    )

    plot_family_metric(
        summary_rows,
        metric="velocity_magnitude_ratio",
        ylabel=(
            "Predicted / true "
            "terrain-effect magnitude"
        ),
        title=(
            "Terrain-induced velocity "
            "magnitude calibration"
        ),
        output_path=(
            output_dir
            /
            "05_velocity_magnitude_by_family.png"
        ),
        ideal_line=1.0,
    )

    plot_family_metric(
        summary_rows,
        metric="direct_bed_cosine",
        ylabel="Cosine similarity",
        title=(
            "Direct bed-force spatial "
            "agreement"
        ),
        output_path=(
            output_dir
            /
            "06_direct_bed_cosine_by_family.png"
        ),
        ideal_line=1.0,
    )

    plot_family_metric(
        summary_rows,
        metric=(
            "direct_bed_magnitude_ratio"
        ),
        ylabel=(
            "Predicted / expected "
            "bed-force magnitude"
        ),
        title=(
            "Direct bed-force magnitude "
            "on unseen terrain"
        ),
        output_path=(
            output_dir
            /
            "07_direct_bed_magnitude_by_family.png"
        ),
        ideal_line=1.0,
    )

    plot_representative_effects(
        data,
        pi_predictions,
        bed_predictions,
        time_index,
        output_dir,
    )

    # ============================================================
    # PRINT SUMMARY
    # ============================================================

    print(
        "\n"
        +
        "=" * 127
    )

    print(
        "OOD TERRAIN GENERALIZATION SUMMARY"
    )

    print(
        "=" * 127
    )

    print(
        f"{'Family':<19}"
        f"{'Model':<13}"
        f"{'Rel L2':>11}"
        f"{'Delta q':>11}"
        f"{'Vel cos':>10}"
        f"{'Vel mag':>10}"
        f"{'Bed cos':>10}"
        f"{'Bed mag':>10}"
    )

    print(
        "-" * 127
    )

    for row in summary_rows:

        print(
            f"{row['family']:<19}"
            f"{row['model']:<13}"
            f"{row['rollout_rel_l2_mean']:>11.4f}"
            f"{row['terrain_delta_q_error_mean']:>11.4f}"
            f"{row['velocity_cosine_mean']:>10.4f}"
            f"{row['velocity_magnitude_ratio_mean']:>10.4f}"
            f"{row['direct_bed_cosine_mean']:>10.4f}"
            f"{row['direct_bed_magnitude_ratio_mean']:>10.4f}"
        )

    print(
        "=" * 127
    )

    print(
        "\nReference:"
    )

    print(
        "  cosine -> 1.0 is better"
    )

    print(
        "  magnitude ratio -> 1.0 is better"
    )

    print(
        "  Rel L2 and Delta q error -> "
        "lower is better"
    )

    print(
        "\nNO OOD DATA WAS USED FOR TRAINING."
    )

    print(
        f"\nSaved results to:\n"
        f"{output_dir}"
    )

    # ============================================================
    # CLEANUP
    # ============================================================

    del pi_state
    del bed_state
    del pi_method
    del bed_method

    gc.collect()

    try:

        jax.clear_caches()

    except AttributeError:

        pass


if __name__ == "__main__":

    main()