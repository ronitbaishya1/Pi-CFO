from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        default=(
            "results/"
            "tangent_random_smooth_geometry_ufno.npz"
        ),
    )

    parser.add_argument(
        "--tan001",
        default=(
            "results/"
            "tangent_after_lamtan_0p001.npz"
        ),
    )

    parser.add_argument(
        "--tan005",
        default=(
            "results/"
            "tangent_after_lamtan_0p005.npz"
        ),
    )

    parser.add_argument(
        "--tan01",
        default=(
            "results/"
            "tangent_after_lamtan_0p01.npz"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/stage_c/figures/tangent"
        ),
    )

    return parser.parse_args()


def get_value(
    archive,
    candidates,
):

    for key in candidates:

        if key in archive:

            value = np.asarray(
                archive[key]
            )

            return float(
                np.mean(value)
            )

    raise KeyError(
        "Could not find any of these keys:\n"
        +
        "\n".join(candidates)
        +
        "\n\nAvailable keys:\n"
        +
        "\n".join(
            archive.files
        )
    )


def load_metrics(
    path,
):

    with np.load(
        path
    ) as archive:

        cosine = get_value(
            archive,
            [
                "smooth_cosine",
                "smooth_cosine_similarity",
                "smooth_tangent_cosine",
            ],
        )

        gain = get_value(
            archive,
            [
                "smooth_gain_ratio",
                "smooth_model_physical_gain_ratio",
                "smooth_tangent_gain_ratio",
            ],
        )

        symmetric = get_value(
            archive,
            [
                "smooth_symmetric_error",
                "smooth_symmetric_tangent_error",
            ],
        )

    return {
        "cosine": cosine,
        "gain": gain,
        "symmetric": symmetric,
    }


def plot_metric(
    lambdas,
    values,
    *,
    ylabel,
    title,
    output,
    reference=None,
):

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.plot(
        lambdas,
        values,
        marker="o",
        linewidth=2,
    )

    if reference is not None:

        ax.axhline(
            reference,
            linestyle="--",
        )

    ax.set_xlabel(
        r"$\lambda_{\mathrm{tan}}$"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    args = parse_args()

    paths = [
        Path(args.baseline),
        Path(args.tan001),
        Path(args.tan005),
        Path(args.tan01),
    ]

    lambdas = np.asarray(
        [
            0.0,
            0.001,
            0.005,
            0.01,
        ]
    )

    metrics = [
        load_metrics(path)
        for path in paths
    ]

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cosine = [
        item["cosine"]
        for item in metrics
    ]

    gain = [
        item["gain"]
        for item in metrics
    ]

    symmetric = [
        item["symmetric"]
        for item in metrics
    ]

    plot_metric(
        lambdas,
        cosine,
        ylabel="Smooth tangent cosine",
        title=(
            "Directional tangent agreement"
        ),
        output=(
            output_dir
            /
            "01_tangent_cosine_vs_lambda.png"
        ),
        reference=1.0,
    )

    plot_metric(
        lambdas,
        gain,
        ylabel=(
            "Model / SWE tangent gain"
        ),
        title=(
            "Tangent magnitude agreement"
        ),
        output=(
            output_dir
            /
            "02_tangent_gain_vs_lambda.png"
        ),
        reference=1.0,
    )

    plot_metric(
        lambdas,
        symmetric,
        ylabel=(
            "Symmetric tangent error"
        ),
        title=(
            "Directional tangent discrepancy"
        ),
        output=(
            output_dir
            /
            "03_tangent_error_vs_lambda.png"
        ),
    )

    print(
        "Saved:",
        output_dir
    )


if __name__ == "__main__":
    main()