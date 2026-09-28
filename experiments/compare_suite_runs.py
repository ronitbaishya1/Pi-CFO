"""
Compare table_08_final_summary.csv from multiple diagnostic runs.

Examples
--------

RK4 comparison:

python experiments/compare_suite_runs.py \
    --entry "RK4-2=results/.../steps2" \
    --entry "RK4-4=results/.../steps4" \
    --entry "RK4-8=results/.../steps8" \
    --output-dir results/.../comparison

Mode comparison:

python experiments/compare_suite_runs.py \
    --entry "Modes-12=results/.../modes12" \
    --entry "Modes-16=results/.../modes16" \
    --output-dir results/.../comparison
"""

from __future__ import annotations

import argparse
import csv

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# METRICS
# =====================================================================

METRICS = [

    "id_rel_l2",

    "id_rmse",

    "id_eta_rel_l2",

    "id_final_mass_drift",

    "gaussian_mean_rel_l2",

    "gaussian_mean_final_mass_drift",

    "gaussian_mean_effect_rel_l2",

    "ood_mean_rel_l2",

    "ood_mean_effect_rel_l2",

    "direct_response_rel_l2",

    "direct_response_cosine",

    "direct_response_gain",
]


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help=(
            "Format: LABEL=PATH. "
            "PATH may be the diagnostic output "
            "directory or table_08_final_summary.csv."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    return parser.parse_args()


# =====================================================================
# ENTRY
# =====================================================================

def parse_entry(
    text,
):

    if "=" not in text:

        raise ValueError(
            "--entry must have format "
            "LABEL=PATH"
        )

    label, path = (
        text.split(
            "=",
            1,
        )
    )

    label = (
        label.strip()
    )

    path = (
        Path(
            path.strip()
        )
        .expanduser()
        .resolve()
    )

    if path.is_dir():

        path = (
            path
            /
            "tables"
            /
            "table_08_final_summary.csv"
        )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    return (
        label,
        path,
    )


# =====================================================================
# LOAD FINAL MODEL ROW
# =====================================================================

def load_final_row(
    path,
):

    with open(
        path,
        "r",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    for row in rows:

        name = (
            row
            .get(
                "model",
                "",
            )
            .lower()
        )

        if (
            "geometry"
            in name
            and
            "ufno"
            in name.replace(
                "-",
                "",
            )
        ):

            return row

    raise KeyError(
        "Could not find Geometry-U-FNO "
        f"row in:\n{path}"
    )


# =====================================================================
# FLOAT
# =====================================================================

def to_float(
    value,
):

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return np.nan


# =====================================================================
# SAVE CSV / MARKDOWN
# =====================================================================

def save_tables(
    rows,
    output_dir,
):

    columns = [
        "experiment",
        *METRICS,
    ]

    csv_path = (
        output_dir
        /
        "comparison_summary.csv"
    )

    with open(
        csv_path,
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
                {
                    column:
                        row.get(
                            column,
                            "",
                        )

                    for column
                    in columns
                }
            )

    md_path = (
        output_dir
        /
        "comparison_summary.md"
    )

    with open(
        md_path,
        "w",
    ) as file:

        file.write(
            "# Comparison summary\n\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                columns
            )
            +
            " |\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                [
                    "---"
                    for _
                    in columns
                ]
            )
            +
            " |\n"
        )

        for row in rows:

            values = []

            for column in columns:

                value = (
                    row.get(
                        column,
                        "",
                    )
                )

                if isinstance(
                    value,
                    float,
                ):

                    values.append(
                        f"{value:.7g}"
                    )

                else:

                    values.append(
                        str(
                            value
                        )
                    )

            file.write(
                "| "
                +
                " | ".join(
                    values
                )
                +
                " |\n"
            )

    return (
        csv_path,
        md_path,
    )


# =====================================================================
# ERROR PLOT
# =====================================================================

def plot_error_metrics(
    rows,
    output_dir,
):

    metrics = [

        "id_rel_l2",

        "gaussian_mean_rel_l2",

        "gaussian_mean_effect_rel_l2",

        "ood_mean_rel_l2",

        "ood_mean_effect_rel_l2",

        "direct_response_rel_l2",
    ]

    labels = [
        row[
            "experiment"
        ]
        for row
        in rows
    ]

    x = np.arange(
        len(
            labels
        )
    )

    width = (
        0.80
        /
        len(
            metrics
        )
    )

    fig = plt.figure(
        figsize=(
            14,
            7,
        )
    )

    ax = fig.add_subplot(
        111
    )

    for index, metric in enumerate(
        metrics
    ):

        offset = (
            index
            -
            (
                len(
                    metrics
                )
                -
                1
            )
            /
            2
        ) * width

        values = [
            row[
                metric
            ]
            for row
            in rows
        ]

        ax.bar(
            x
            +
            offset,
            values,
            width,
            label=metric,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "Geometry-U-FNO diagnostic comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
    )

    fig.tight_layout()

    path = (
        output_dir
        /
        "01_error_comparison.png"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# =====================================================================
# DIRECT RESPONSE PLOT
# =====================================================================

def plot_direct_response(
    rows,
    output_dir,
):

    labels = [
        row[
            "experiment"
        ]
        for row
        in rows
    ]

    cosine = np.asarray(
        [
            row[
                "direct_response_cosine"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    gain = np.asarray(
        [
            row[
                "direct_response_gain"
            ]
            for row
            in rows
        ],
        dtype=np.float64,
    )

    x = np.arange(
        len(
            rows
        )
    )

    width = 0.35

    fig = plt.figure(
        figsize=(
            10,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.bar(
        x
        -
        width
        /
        2,
        cosine,
        width,
        label="Direct-response cosine",
    )

    ax.bar(
        x
        +
        width
        /
        2,
        gain,
        width,
        label="Direct-response gain",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="Ideal",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Value"
    )

    ax.set_title(
        "Direct bathymetry-response comparison"
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "02_direct_response_comparison.png"
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    output_dir = (
        Path(
            args.output_dir
        )
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for entry in args.entry:

        (
            label,
            path,
        ) = parse_entry(
            entry
        )

        source = load_final_row(
            path
        )

        row = {

            "experiment":
                label,
        }

        for metric in METRICS:

            row[
                metric
            ] = to_float(
                source.get(
                    metric,
                    np.nan,
                )
            )

        rows.append(
            row
        )

    (
        csv_path,
        md_path,
    ) = save_tables(
        rows,
        output_dir,
    )

    error_plot = (
        plot_error_metrics(
            rows,
            output_dir,
        )
    )

    direct_plot = (
        plot_direct_response(
            rows,
            output_dir,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "COMPARISON SUMMARY"
    )

    print(
        "=" * 80
    )

    for row in rows:

        print()
        print(
            row[
                "experiment"
            ]
        )

        for metric in METRICS:

            print(
                f"  {metric:38s}: "
                f"{row[metric]:.8f}"
            )

    print()
    print(
        "CSV:",
        csv_path,
    )

    print(
        "Markdown:",
        md_path,
    )

    print(
        "Error plot:",
        error_plot,
    )

    print(
        "Direct-response plot:",
        direct_plot,
    )


if __name__ == "__main__":

    main()