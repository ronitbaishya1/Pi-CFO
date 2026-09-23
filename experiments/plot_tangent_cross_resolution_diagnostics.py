"""
Cross-resolution tangent Geometry-U-FNO diagnostics.

Comparison:
    PyClaw 64
    Geometry-U-FNO 32->64
    Tangent U-FNO 0.001 32->64
    Tangent U-FNO 0.005 32->64
    Tangent U-FNO 0.01 32->64
    Geometry-U-FNO 64-trained

Outputs:
    01_direct_condition_resolution.png
    02_isolated_hill_effect_resolution.png
    03_terrain_speed_gaussian_resolution.png
    04_terrain_speed_unseen_resolution.png
    05_cross_sections_resolution.png
    06_3d_gaussian_resolution.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


from models.geometry_ufno import (
    GeometryUFNO2d,
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

import experiments.plot_wb_bed_diagnostics as diag


RESOLUTION = 64

DX = 5.0 / 64.0
DY = 5.0 / 64.0

GRAVITY = 1.0


MODEL_KEYS = [
    "base32",
    "tan001",
    "tan005",
    "tan01",
    "trained64",
]


MODEL_LABELS = {

    "base32":
        "Geo-U-FNO 32→64",

    "tan001":
        r"Tangent .001 32→64",

    "tan005":
        r"Tangent .005 32→64",

    "tan01":
        r"Tangent .01 32→64",

    "trained64":
        "Geo-U-FNO 64-trained",
}


DEFAULT_CHECKPOINTS = {

    "base32":
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

    "trained64":
        (
            "checkpoints/"
            "geometry_wb_bed_pi/"
            "geometry_ufno/"
            "res64/"
            "lamwb_0p1/"
            "seed0"
        ),
}


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--counterfactual-data",
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--ood-data",
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_64.h5"
        ),
    )

    parser.add_argument(
        "--base32-checkpoint",
        default=DEFAULT_CHECKPOINTS[
            "base32"
        ],
    )

    parser.add_argument(
        "--tan001-checkpoint",
        default=DEFAULT_CHECKPOINTS[
            "tan001"
        ],
    )

    parser.add_argument(
        "--tan005-checkpoint",
        default=DEFAULT_CHECKPOINTS[
            "tan005"
        ],
    )

    parser.add_argument(
        "--tan01-checkpoint",
        default=DEFAULT_CHECKPOINTS[
            "tan01"
        ],
    )

    parser.add_argument(
        "--trained64-checkpoint",
        default=DEFAULT_CHECKPOINTS[
            "trained64"
        ],
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--direct-case",
        default="hill_right",
    )

    parser.add_argument(
        "--cross-section-case",
        default="hill_center",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "tangent_cross_resolution"
        ),
    )

    return parser.parse_args()


def resolve_path(
    value,
):

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def configure_diag():

    diag.NX = 64
    diag.NY = 64

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = -2.5
    diag.X_MAX = 2.5

    diag.Y_MIN = -2.5
    diag.Y_MAX = 2.5


def build_method():

    model = GeometryUFNO2d(
        num_channels=3,

        modes1=12,
        modes2=12,

        width=64,

        num_blocks=4,
        num_u_blocks=2,

        geometry_width=16,
        geometry_depth=2,

        dx=DX,
        dy=DY,

        include_gradient_magnitude=False,

        use_time=True,
    )

    return (
        WellBalancedBathymetryBedPICFO(
            model=model,

            input_shape=(
                64,
                64,
                3,
            ),

            condition_shape=(
                64,
                64,
                1,
            ),

            gamma=1.0e-5,

            spline_type="quintic",

            lambda_pde=0.03,
            lambda_bed=0.70,
            lambda_wb=0.10,

            wb_eta0=1.5,

            dx=DX,
            dy=DY,

            gravity=GRAVITY,
        )
    )


def restore(
    method,
    root,
):

    root = resolve_path(
        root
    )

    target = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target,
        ckpt_dir=str(root),
        prefix="best",
        step=None,
        max_to_keep=1,
    )


def build_all(
    args,
):

    checkpoint_paths = {

        "base32":
            args.base32_checkpoint,

        "tan001":
            args.tan001_checkpoint,

        "tan005":
            args.tan005_checkpoint,

        "tan01":
            args.tan01_checkpoint,

        "trained64":
            args.trained64_checkpoint,
    }

    methods = {}
    states = {}

    for key in MODEL_KEYS:

        print(
            "Restoring",
            MODEL_LABELS[key],
        )

        method = build_method()

        state = restore(
            method,
            checkpoint_paths[key],
        )

        methods[key] = method
        states[key] = state

    return methods, states


def evaluate(
    archive,
    case_name,
    *,
    methods,
    states,
    sample_index,
    steps_per_segment,
):

    data = archive.get_pair(
        case_name,
        sample_index=sample_index,
    )

    result = {
        "data": data,
        "models": {},
    }

    for key in MODEL_KEYS:

        hill, flat = (
            diag.model_pair_rollout(
                methods[key],
                states[key],
                data,
                steps_per_segment=(
                    steps_per_segment
                ),
            )
        )

        result[
            "models"
        ][key] = {
            "hill": hill,
            "flat": flat,
        }

    return result


def direct_condition(
    result,
    methods,
    states,
    *,
    plot_time,
    output,
):

    data = result[
        "data"
    ]

    time = data[
        "time"
    ]

    index = int(
        np.argmin(
            np.abs(
                time
                -
                plot_time
            )
        )
    )

    t_value = float(
        time[index]
    )

    q = data[
        "terrain_q"
    ][index]

    b = data["b"]

    flat_b = np.zeros_like(
        b
    )

    h = np.maximum(
        q[..., 0],
        1.0e-6,
    )

    expected_hu = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_x(
            b
        )
    )

    expected_hv = (
        -GRAVITY
        *
        h
        *
        diag.central_diff_y(
            b
        )
    )

    responses = {}

    for key in MODEL_KEYS:

        with_b = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q,
                b,
                t_value,
            )
        )

        no_b = (
            diag.model_vector_field(
                methods[key],
                states[key],
                q,
                flat_b,
                t_value,
            )
        )

        responses[key] = (
            with_b
            -
            no_b
        )

    hu = [
        expected_hu,
        *[
            responses[key][..., 1]
            for key in MODEL_KEYS
        ],
    ]

    hv = [
        expected_hv,
        *[
            responses[key][..., 2]
            for key in MODEL_KEYS
        ],
    ]

    labels = [
        "SWE expected",
        *[
            MODEL_LABELS[key]
            for key in MODEL_KEYS
        ],
    ]

    hu_limit = max(
        max(
            np.max(
                np.abs(field)
            )
            for field in hu
        ),
        1.0e-8,
    )

    hv_limit = max(
        max(
            np.max(
                np.abs(field)
            )
            for field in hv
        ),
        1.0e-8,
    )

    columns = len(
        labels
    )

    fig, axes = plt.subplots(
        2,
        columns,
        figsize=(
            4.3 * columns,
            8.5,
        ),
    )

    for column in range(
        columns
    ):

        image = diag.heatmap(
            axes[
                0,
                column
            ],
            hu[column],
            vlim=hu_limit,
            title=(
                labels[column]
                +
                "\n"
                +
                r"$\Delta(hu)_t$"
            ),
            b=b,
        )

        fig.colorbar(
            image,
            ax=axes[
                0,
                column
            ],
            shrink=0.78,
        )

        image = diag.heatmap(
            axes[
                1,
                column
            ],
            hv[column],
            vlim=hv_limit,
            title=(
                labels[column]
                +
                "\n"
                +
                r"$\Delta(hv)_t$"
            ),
            b=b,
        )

        fig.colorbar(
            image,
            ax=axes[
                1,
                column
            ],
            shrink=0.78,
        )

    fig.suptitle(
        (
            "64×64 direct-condition resolution test "
            f"at t={t_value:.2f}"
        ),
        fontsize=18,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def isolated_error(
    results,
    output,
):

    cases = [
        "hill_left",
        "hill_center",
        "hill_right",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
    )

    for column, case_name in enumerate(
        cases
    ):

        result = results[
            case_name
        ]

        data = result[
            "data"
        ]

        time = data[
            "time"
        ]

        ax = axes[column]

        for key in MODEL_KEYS:

            model = result[
                "models"
            ][key]

            error = (
                diag.relative_effect_error(
                    data["terrain_q"],
                    data["flat_q"],
                    model["hill"],
                    model["flat"],
                )
            )

            ax.plot(
                time,
                error,
                linewidth=2,
                label=MODEL_LABELS[key],
            )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Time"
        )

        ax.set_ylabel(
            "Relative terrain-effect error"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend(
            fontsize=7
        )

    fig.suptitle(
        (
            "64×64 isolated Gaussian-hill "
            "effect error"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def speed_grid(
    results,
    case_names,
    *,
    plot_time,
    title,
    output,
):

    rows = len(
        case_names
    )

    columns = (
        1
        +
        len(
            MODEL_KEYS
        )
    )

    labels = [
        "PyClaw 64",
        *[
            MODEL_LABELS[key]
            for key in MODEL_KEYS
        ],
    ]

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            4.2 * columns,
            4.2 * rows,
        ),
    )

    if rows == 1:

        axes = np.asarray(
            axes
        )[
            None,
            :
        ]

    for row, case_name in enumerate(
        case_names
    ):

        result = results[
            case_name
        ]

        data = result[
            "data"
        ]

        index = int(
            np.argmin(
                np.abs(
                    data["time"]
                    -
                    plot_time
                )
            )
        )

        fields = [
            diag.terrain_speed_effect(
                data["terrain_q"],
                data["flat_q"],
                index,
            )
        ]

        for key in MODEL_KEYS:

            model = result[
                "models"
            ][key]

            fields.append(
                diag.terrain_speed_effect(
                    model["hill"],
                    model["flat"],
                    index,
                )
            )

        limit = max(
            max(
                np.max(
                    np.abs(field)
                )
                for field in fields
            ),
            1.0e-8,
        )

        for column in range(
            columns
        ):

            image = diag.heatmap(
                axes[
                    row,
                    column
                ],
                fields[column],
                vlim=limit,
                title=(
                    case_name
                    +
                    "\n"
                    +
                    labels[column]
                ),
                b=data["b"],
            )

            fig.colorbar(
                image,
                ax=axes[
                    row,
                    column
                ],
                shrink=0.75,
            )

    fig.suptitle(
        (
            title
            +
            f" at t={plot_time:.2f}"
        ),
        fontsize=18,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def cell_coordinates():

    x = (
        -2.5
        +
        (
            np.arange(64)
            +
            0.5
        )
        *
        DX
    )

    y = (
        -2.5
        +
        (
            np.arange(64)
            +
            0.5
        )
        *
        DY
    )

    return x, y


def cross_sections(
    result,
    *,
    plot_time,
    output,
):

    data = result[
        "data"
    ]

    index = int(
        np.argmin(
            np.abs(
                data["time"]
                -
                plot_time
            )
        )
    )

    b = data["b"]

    max_index = np.unravel_index(
        np.argmax(b),
        b.shape,
    )

    ix = int(
        max_index[0]
    )

    iy = int(
        max_index[1]
    )

    x, y = cell_coordinates()

    truth = data[
        "terrain_q"
    ][index]

    variables = [
        ("h", 0),
        ("hu", 1),
        ("hv", 2),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(16, 9),
    )

    for column, (
        name,
        channel,
    ) in enumerate(
        variables
    ):

        ax = axes[
            0,
            column
        ]

        bathy_axis = (
            ax.twinx()
        )

        ax.plot(
            x,
            truth[
                :,
                iy,
                channel
            ],
            linewidth=2.4,
            label="PyClaw 64",
        )

        for key in MODEL_KEYS:

            field = (
                result[
                    "models"
                ][key][
                    "hill"
                ][index]
            )

            ax.plot(
                x,
                field[
                    :,
                    iy,
                    channel
                ],
                linewidth=1.5,
                label=MODEL_LABELS[key],
            )

        bathy_axis.plot(
            x,
            b[
                :,
                iy
            ],
            linestyle="--",
            alpha=0.6,
        )

        ax.set_title(
            (
                f"{name} along "
                f"y={y[iy]:.3f}"
            )
        )

        ax.set_xlabel("x")
        ax.set_ylabel(name)
        bathy_axis.set_ylabel("b")

        ax.grid(
            alpha=0.3
        )

        if column == 0:

            ax.legend(
                fontsize=6,
                loc="best",
            )

        ax = axes[
            1,
            column
        ]

        bathy_axis = (
            ax.twinx()
        )

        ax.plot(
            y,
            truth[
                ix,
                :,
                channel
            ],
            linewidth=2.4,
            label="PyClaw 64",
        )

        for key in MODEL_KEYS:

            field = (
                result[
                    "models"
                ][key][
                    "hill"
                ][index]
            )

            ax.plot(
                y,
                field[
                    ix,
                    :,
                    channel
                ],
                linewidth=1.5,
            )

        bathy_axis.plot(
            y,
            b[
                ix,
                :
            ],
            linestyle="--",
            alpha=0.6,
        )

        ax.set_title(
            (
                f"{name} along "
                f"x={x[ix]:.3f}"
            )
        )

        ax.set_xlabel("y")
        ax.set_ylabel(name)
        bathy_axis.set_ylabel("b")

        ax.grid(
            alpha=0.3
        )

    fig.suptitle(
        (
            "64×64 Gaussian-hill "
            "cross-resolution cross-sections"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    fig.savefig(
        output,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_3d(
    result,
    *,
    plot_time,
    output,
):

    data = result[
        "data"
    ]

    index = int(
        np.argmin(
            np.abs(
                data["time"]
                -
                plot_time
            )
        )
    )

    b = data["b"]

    x, y = cell_coordinates()

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    surfaces = [

        (
            "PyClaw 64",
            data[
                "terrain_q"
            ][
                index,
                ...,
                0
            ]
            +
            b,
        )
    ]

    for key in MODEL_KEYS:

        q = (
            result[
                "models"
            ][key][
                "hill"
            ][index]
        )

        surfaces.append(
            (
                MODEL_LABELS[key],
                q[
                    ...,
                    0
                ]
                +
                b,
            )
        )

    columns = len(
        surfaces
    )

    fig = plt.figure(
        figsize=(
            5 * columns,
            5,
        )
    )

    for column, (
        label,
        eta,
    ) in enumerate(
        surfaces,
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            columns,
            column,
            projection="3d",
        )

        ax.plot_surface(
            X,
            Y,
            b,
            alpha=0.25,
            linewidth=0,
        )

        ax.plot_surface(
            X,
            Y,
            eta,
            alpha=0.82,
            linewidth=0,
        )

        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(
            r"$\eta$"
        )

    fig.suptitle(
        (
            "64×64 cross-resolution "
            "3D Gaussian-hill response"
        ),
        fontsize=17,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    args = parse_args()

    configure_diag()

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    methods, states = build_all(
        args
    )

    gaussian_archive = (
        diag.H5Archive(
            args.counterfactual_data
        )
    )

    ood_archive = (
        diag.H5Archive(
            args.ood_data
        )
    )

    try:

        gaussian_results = {}

        for case_name in [
            "hill_left",
            "hill_center",
            "hill_right",
        ]:

            gaussian_results[
                case_name
            ] = evaluate(
                gaussian_archive,
                case_name,
                methods=methods,
                states=states,
                sample_index=(
                    args.sample_index
                ),
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        ood_results = {}

        for case_name in [
            "two_hills",
            "narrow_tall",
            "elongated_ridge",
            "rotated_ridge",
            "multi_hill",
        ]:

            ood_results[
                case_name
            ] = evaluate(
                ood_archive,
                case_name,
                methods=methods,
                states=states,
                sample_index=(
                    args.sample_index
                ),
                steps_per_segment=(
                    args.steps_per_segment
                ),
            )

        direct_condition(
            gaussian_results[
                args.direct_case
            ],
            methods,
            states,
            plot_time=(
                args.plot_time
            ),
            output=(
                output_dir
                /
                "01_direct_condition_resolution.png"
            ),
        )

        isolated_error(
            gaussian_results,
            output=(
                output_dir
                /
                "02_isolated_hill_effect_resolution.png"
            ),
        )

        speed_grid(
            gaussian_results,
            [
                "hill_left",
                "hill_center",
                "hill_right",
            ],
            plot_time=(
                args.plot_time
            ),
            title=(
                "64×64 terrain-induced speed "
                "for Gaussian hills"
            ),
            output=(
                output_dir
                /
                "03_terrain_speed_gaussian_resolution.png"
            ),
        )

        speed_grid(
            ood_results,
            [
                "two_hills",
                "narrow_tall",
                "elongated_ridge",
                "rotated_ridge",
                "multi_hill",
            ],
            plot_time=(
                args.plot_time
            ),
            title=(
                "64×64 terrain-induced speed "
                "on unseen bathymetry"
            ),
            output=(
                output_dir
                /
                "04_terrain_speed_unseen_resolution.png"
            ),
        )

        cross_result = (
            gaussian_results[
                args.cross_section_case
            ]
        )

        cross_sections(
            cross_result,
            plot_time=(
                args.plot_time
            ),
            output=(
                output_dir
                /
                "05_cross_sections_resolution.png"
            ),
        )

        plot_3d(
            cross_result,
            plot_time=(
                args.plot_time
            ),
            output=(
                output_dir
                /
                "06_3d_gaussian_resolution.png"
            ),
        )

    finally:

        gaussian_archive.close()
        ood_archive.close()

    print()
    print(
        "Saved:",
        output_dir
    )


if __name__ == "__main__":
    main()