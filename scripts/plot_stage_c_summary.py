from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        "--root",
        default="results/stage_c",
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/stage_c/"
            "figures/summary"
        ),
    )

    return parser.parse_args()


def load_scenario(
    root,
    scenario,
):

    values = {}

    for variant in VARIANTS:

        path = (
            Path(root)
            /
            scenario
            /
            variant
            /
            "metrics.npz"
        )

        values[
            variant
        ] = np.load(path)

    return values


def bar_plot(
    values,
    *,
    ylabel,
    title,
    output,
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = np.arange(
        len(VARIANTS)
    )

    ax.bar(
        x,
        values,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        LABELS,
        rotation=15,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def time_plot(
    metrics,
    key,
    *,
    ylabel,
    title,
    output,
    training_boundary=False,
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for variant, label in zip(
        VARIANTS,
        LABELS,
    ):

        archive = metrics[
            variant
        ]

        ax.plot(
            archive["time"],
            archive[key],
            linewidth=2,
            label=label,
        )

    if training_boundary:

        ax.axvline(
            1.0,
            linestyle="--",
            label="Training horizon",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def grouped_generalization(
    scenarios,
    output,
):

    scenario_names = [
        "ID 32",
        "OOD 32",
        "Perturbed 32",
        "32→64",
        "32→64 OOD",
    ]

    keys = [
        "id32",
        "ood32",
        "perturbed32",
        "cross64",
        "cross64_ood",
    ]

    x = np.arange(
        len(keys)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    for model_index, (
        variant,
        label,
    ) in enumerate(
        zip(
            VARIANTS,
            LABELS,
        )
    ):

        values = [
            float(
                scenarios[
                    scenario
                ][
                    variant
                ][
                    "global_relative_l2"
                ]
            )
            for scenario in keys
        ]

        ax.bar(
            x
            +
            (
                model_index
                -
                1.5
            )
            *
            width,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        scenario_names
    )

    ax.set_ylabel(
        "Relative L2 error"
    )

    ax.set_title(
        "Generalization across evaluation regimes"
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def channel_plot(
    scenario,
    *,
    title,
    output,
):

    keys = [
        "h_relative_l2",
        "hu_relative_l2",
        "hv_relative_l2",
        "eta_relative_l2",
    ]

    names = [
        "h",
        "hu",
        "hv",
        r"$\eta$",
    ]

    x = np.arange(
        len(keys)
    )

    width = 0.18

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for model_index, (
        variant,
        label,
    ) in enumerate(
        zip(
            VARIANTS,
            LABELS,
        )
    ):

        values = [
            float(
                scenario[
                    variant
                ][key]
            )
            for key in keys
        ]

        ax.bar(
            x
            +
            (
                model_index
                -
                1.5
            )
            *
            width,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names)

    ax.set_ylabel(
        "Relative L2 error"
    )

    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend()

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

    scenarios = {

        "id32":
            load_scenario(
                args.root,
                "id32",
            ),

        "ood32":
            load_scenario(
                args.root,
                "ood32",
            ),

        "perturbed32":
            load_scenario(
                args.root,
                "perturbed32",
            ),

        "long32_t2":
            load_scenario(
                args.root,
                "long32_t2",
            ),

        "cross64":
            load_scenario(
                args.root,
                "cross64",
            ),

        "cross64_ood":
            load_scenario(
                args.root,
                "cross64_ood",
            ),
    }

    for scenario, title in [
        (
            "id32",
            "32×32 in-distribution",
        ),
        (
            "ood32",
            "32×32 OOD",
        ),
        (
            "perturbed32",
            "Perturbed initial condition",
        ),
        (
            "cross64",
            "Zero-shot 32×32 → 64×64",
        ),
        (
            "cross64_ood",
            "Zero-shot 64×64 OOD",
        ),
    ]:

        bar_plot(
            [
                float(
                    scenarios[
                        scenario
                    ][
                        variant
                    ][
                        "global_relative_l2"
                    ]
                )
                for variant
                in VARIANTS
            ],
            ylabel="Relative L2 error",
            title=title,
            output=(
                output_dir
                /
                f"{scenario}_relative_l2.png"
            ),
        )

    time_plot(
        scenarios[
            "long32_t2"
        ],
        "error_by_time",
        ylabel="Relative L2 error",
        title="Long-horizon rollout",
        output=(
            output_dir
            /
            "long_rollout_error.png"
        ),
        training_boundary=True,
    )

    time_plot(
        scenarios[
            "long32_t2"
        ],
        "eta_error_by_time",
        ylabel=(
            "Free-surface relative L2"
        ),
        title=(
            "Long-horizon free-surface error"
        ),
        output=(
            output_dir
            /
            "long_eta_error.png"
        ),
        training_boundary=True,
    )

    time_plot(
        scenarios[
            "long32_t2"
        ],
        "mass_relative_error",
        ylabel="Relative mass error",
        title="Long-horizon mass error",
        output=(
            output_dir
            /
            "long_mass_error.png"
        ),
        training_boundary=True,
    )

    time_plot(
        scenarios[
            "ood32"
        ],
        "error_by_time",
        ylabel="Relative L2 error",
        title="OOD rollout error",
        output=(
            output_dir
            /
            "ood32_error_vs_time.png"
        ),
    )

    channel_plot(
        scenarios[
            "ood32"
        ],
        title=(
            "OOD error by physical variable"
        ),
        output=(
            output_dir
            /
            "ood32_channel_errors.png"
        ),
    )

    channel_plot(
        scenarios[
            "cross64"
        ],
        title=(
            "32→64 error by physical variable"
        ),
        output=(
            output_dir
            /
            "cross64_channel_errors.png"
        ),
    )

    grouped_generalization(
        scenarios,
        output=(
            output_dir
            /
            "generalization_summary.png"
        ),
    )

    print(
        "Saved:",
        output_dir
    )


if __name__ == "__main__":
    main()