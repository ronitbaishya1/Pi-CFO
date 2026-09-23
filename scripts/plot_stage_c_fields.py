from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.bathy_data import (
    load_bathymetry_dataset,
    require_split,
)


VARIANTS = [
    "baseline",
    "tan001",
    "tan005",
    "tan01",
]

LABELS = [
    "Baseline",
    r"$\lambda_{tan}=0.001$",
    r"$\lambda_{tan}=0.005$",
    r"$\lambda_{tan}=0.01$",
]


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-path",
        required=True,
    )

    parser.add_argument(
        "--split",
        default="test_id",
    )

    parser.add_argument(
        "--result-root",
        required=True,
    )

    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def plot_field(
    truth,
    predictions,
    *,
    title,
    output,
):

    all_fields = [
        truth,
        *predictions,
    ]

    vmin = min(
        float(np.min(field))
        for field in all_fields
    )

    vmax = max(
        float(np.max(field))
        for field in all_fields
    )

    errors = [
        np.abs(
            prediction
            -
            truth
        )
        for prediction
        in predictions
    ]

    error_max = max(
        float(
            np.max(error)
        )
        for error in errors
    )

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 7),
    )

    panels = [
        ("Truth", truth),
        *list(
            zip(
                LABELS,
                predictions,
            )
        ),
    ]

    for index, (
        label,
        field,
    ) in enumerate(panels):

        image = axes[
            0,
            index
        ].imshow(
            field.T,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        axes[
            0,
            index
        ].set_title(label)

        fig.colorbar(
            image,
            ax=axes[
                0,
                index
            ],
            shrink=0.75,
        )

    axes[
        1,
        0
    ].axis("off")

    for index, (
        label,
        error,
    ) in enumerate(
        zip(
            LABELS,
            errors,
        ),
        start=1,
    ):

        image = axes[
            1,
            index
        ].imshow(
            error.T,
            origin="lower",
            vmin=0.0,
            vmax=error_max,
        )

        axes[
            1,
            index
        ].set_title(
            label
            +
            " absolute error"
        )

        fig.colorbar(
            image,
            ax=axes[
                1,
                index
            ],
            shrink=0.75,
        )

    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_bathymetry_dataset(
        Path(
            args.dataset_path
        )
    )

    split = require_split(
        dataset,
        args.split,
    )

    q = np.asarray(
        split["q"]
    )

    b = np.asarray(
        split["bathymetry"]
    )

    trajectory = (
        args.trajectory_index
    )

    time_index = (
        args.time_index
    )

    truth = q[
        trajectory,
        time_index,
    ]

    bathymetry = b[
        trajectory,
        ...,
        0,
    ]

    predictions = []

    for variant in VARIANTS:

        path = (
            Path(
                args.result_root
            )
            /
            variant
            /
            "prediction.npy"
        )

        prediction = np.load(
            path
        )

        predictions.append(
            prediction[
                trajectory,
                time_index,
            ]
        )

    fields = {

        "h":
            (
                truth[..., 0],
                [
                    prediction[..., 0]
                    for prediction
                    in predictions
                ],
            ),

        "hu":
            (
                truth[..., 1],
                [
                    prediction[..., 1]
                    for prediction
                    in predictions
                ],
            ),

        "hv":
            (
                truth[..., 2],
                [
                    prediction[..., 2]
                    for prediction
                    in predictions
                ],
            ),

        "eta":
            (
                truth[..., 0]
                +
                bathymetry,

                [
                    prediction[..., 0]
                    +
                    bathymetry

                    for prediction
                    in predictions
                ],
            ),
    }

    for name, (
        truth_field,
        prediction_fields,
    ) in fields.items():

        plot_field(
            truth_field,
            prediction_fields,
            title=name,
            output=(
                output_dir
                /
                f"{name}.png"
            ),
        )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        bathymetry.T,
        origin="lower",
    )

    ax.set_title(
        "Bathymetry"
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        /
        "bathymetry.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_dir
    )


if __name__ == "__main__":
    main()