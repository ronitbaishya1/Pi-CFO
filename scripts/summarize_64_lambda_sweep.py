"""
Summarize Geometry-U-FNO lambda sweeps.

Reads each run's:

    metrics.npz

and optionally:

    diagnostics/tables/table_08_final_summary.csv

Outputs:

    lambda_sweep_summary.csv
    lambda_sweep_summary.md

    01_validation_history.png
    02_weighted_loss_balance.png
    03_id_metrics.png

The important quantities include BOTH raw and weighted losses:

    L_CFO
    lambda_PDE * L_PDE
    lambda_bed * L_bed
    lambda_WB * L_WB

because changes in grid resolution can change the raw magnitudes
of derivative-based PDE and bed losses.
"""

from __future__ import annotations

import argparse
import csv

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--extra-metrics",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional additional metrics.npz files, "
            "for example the original 64x64 "
            "(0.03,0.70,0.10) run."
        ),
    )

    return parser.parse_args()


# =====================================================================
# HELPERS
# =====================================================================

def scalar(
    data,
    key,
    default=np.nan,
):

    if key not in data:
        return default

    value = np.asarray(
        data[
            key
        ]
    )

    if value.size == 0:
        return default

    return float(
        value.reshape(
            -1
        )[0]
    )


def tail_median(
    array,
    fraction,
):

    array = np.asarray(
        array,
        dtype=np.float64,
    ).reshape(
        -1
    )

    if array.size == 0:
        return np.nan

    number = max(
        20,
        int(
            np.ceil(
                array.size
                *
                fraction
            )
        ),
    )

    number = min(
        number,
        array.size,
    )

    return float(
        np.median(
            array[
                -number:
            ]
        )
    )


def run_name_from_path(
    metrics_path,
):

    # metrics.npz
    #   parent      = seed0
    #   parent.parent = run tag

    parent = (
        metrics_path
        .parent
    )

    if (
        parent.name
        .startswith(
            "seed"
        )
    ):

        return (
            parent
            .parent
            .name
        )

    return parent.name


def load_diagnostic_summary(
    metrics_path,
):

    diagnostics_csv = (
        metrics_path
        .parent
        /
        "diagnostics"
        /
        "tables"
        /
        "table_08_final_summary.csv"
    )

    if not diagnostics_csv.exists():

        return {}

    with open(
        diagnostics_csv,
        "r",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:
        return {}

    selected = None

    for row in rows:

        model_name = (
            row
            .get(
                "model",
                "",
            )
            .lower()
        )

        if (
            "geometry"
            in model_name
            and
            "ufno"
            in model_name
            .replace(
                "-",
                "",
            )
        ):

            selected = row
            break

    if selected is None:

        # Fall back to the final row if model naming differs.
        selected = rows[
            -1
        ]

    result = {}

    for key, value in selected.items():

        if key is None:
            continue

        column = (
            "diag_"
            +
            str(
                key
            )
        )

        try:

            result[
                column
            ] = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            result[
                column
            ] = value

    return result


# =====================================================================
# LOAD ONE METRICS FILE
# =====================================================================

def load_run(
    metrics_path,
    tail_fraction,
):

    with np.load(
        metrics_path,
        allow_pickle=True,
    ) as data:

        lambda_pde = scalar(
            data,
            "lambda_pde",
        )

        lambda_bed = scalar(
            data,
            "lambda_bed",
        )

        lambda_wb = scalar(
            data,
            "lambda_wb",
        )

        cfo_tail = tail_median(
            data[
                "cfo_loss"
            ],
            tail_fraction,
        )

        pde_tail = tail_median(
            data[
                "pde_loss"
            ],
            tail_fraction,
        )

        bed_tail = tail_median(
            data[
                "bed_loss"
            ],
            tail_fraction,
        )

        wb_tail = tail_median(
            data[
                "wb_loss"
            ],
            tail_fraction,
        )

        weighted_pde = (
            lambda_pde
            *
            pde_tail
        )

        weighted_bed = (
            lambda_bed
            *
            bed_tail
        )

        weighted_wb = (
            lambda_wb
            *
            wb_tail
        )

        cfo_safe = max(
            abs(
                cfo_tail
            ),
            1.0e-15,
        )

        row = {

            "run":
                run_name_from_path(
                    metrics_path
                ),

            "metrics_path":
                str(
                    metrics_path
                ),

            "resolution":
                scalar(
                    data,
                    "resolution",
                ),

            "lambda_pde":
                lambda_pde,

            "lambda_bed":
                lambda_bed,

            "lambda_wb":
                lambda_wb,

            "best_epoch":
                scalar(
                    data,
                    "best_epoch",
                ),

            "best_validation_l2":
                scalar(
                    data,
                    "best_validation_l2",
                ),

            "test_relative_l2":
                scalar(
                    data,
                    "test_relative_l2",
                ),

            "test_rmse":
                scalar(
                    data,
                    "test_rmse",
                ),

            "test_relative_frobenius":
                scalar(
                    data,
                    "test_relative_frobenius",
                ),

            "tail_cfo_loss":
                cfo_tail,

            "tail_pde_loss":
                pde_tail,

            "tail_bed_loss":
                bed_tail,

            "tail_wb_loss":
                wb_tail,

            "weighted_pde":
                weighted_pde,

            "weighted_bed":
                weighted_bed,

            "weighted_wb":
                weighted_wb,

            "weighted_pde_over_cfo":
                weighted_pde
                /
                cfo_safe,

            "weighted_bed_over_cfo":
                weighted_bed
                /
                cfo_safe,

            "weighted_wb_over_cfo":
                weighted_wb
                /
                cfo_safe,

            "validation_epochs":
                np.asarray(
                    data[
                        "validation_epochs"
                    ]
                ),

            "validation_l2":
                np.asarray(
                    data[
                        "validation_l2"
                    ]
                ),
        }

    row.update(
        load_diagnostic_summary(
            metrics_path
        )
    )

    return row


# =====================================================================
# CSV SAFE ROW
# =====================================================================

def serializable_row(
    row,
):

    result = {}

    for key, value in row.items():

        if isinstance(
            value,
            np.ndarray,
        ):
            continue

        result[
            key
        ] = value

    return result


# =====================================================================
# SAVE TABLES
# =====================================================================

def save_tables(
    rows,
    output_dir,
):

    clean_rows = [
        serializable_row(
            row
        )
        for row
        in rows
    ]

    all_columns = []

    for row in clean_rows:

        for key in row.keys():

            if key not in all_columns:
                all_columns.append(
                    key
                )

    csv_path = (
        output_dir
        /
        "lambda_sweep_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=all_columns,
        )

        writer.writeheader()

        writer.writerows(
            clean_rows
        )

    md_path = (
        output_dir
        /
        "lambda_sweep_summary.md"
    )

    core_columns = [
        "run",
        "lambda_pde",
        "lambda_bed",
        "lambda_wb",
        "best_epoch",
        "best_validation_l2",
        "test_relative_l2",
        "test_rmse",
        "weighted_pde_over_cfo",
        "weighted_bed_over_cfo",
        "weighted_wb_over_cfo",
    ]

    with open(
        md_path,
        "w",
    ) as file:

        file.write(
            "# 64x64 lambda sweep\n\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                core_columns
            )
            +
            " |\n"
        )

        file.write(
            "| "
            +
            " | ".join(
                "---"
                for _
                in core_columns
            )
            +
            " |\n"
        )

        for row in clean_rows:

            values = []

            for column in core_columns:

                value = row.get(
                    column,
                    "",
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
# VALIDATION HISTORY
# =====================================================================

def plot_validation_history(
    rows,
    output_dir,
):

    fig = plt.figure(
        figsize=(
            10,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    for row in rows:

        epochs = row[
            "validation_epochs"
        ]

        error = row[
            "validation_l2"
        ]

        if (
            len(
                epochs
            )
            ==
            0
        ):

            continue

        label = (
            f"PDE={row['lambda_pde']:g}, "
            f"bed={row['lambda_bed']:g}, "
            f"WB={row['lambda_wb']:g}"
        )

        ax.plot(
            epochs,
            error,
            marker="o",
            label=label,
        )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Validation relative $L_2$"
    )

    ax.set_title(
        "64x64 Geometry-U-FNO validation histories"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir
        /
        "01_validation_history.png"
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
# WEIGHTED LOSS BALANCE
# =====================================================================

def plot_weighted_loss_balance(
    rows,
    output_dir,
):

    labels = [
        (
            f"PDE={row['lambda_pde']:g}\n"
            f"bed={row['lambda_bed']:g}\n"
            f"WB={row['lambda_wb']:g}"
        )
        for row
        in rows
    ]

    x = np.arange(
        len(
            rows
        )
    )

    width = 0.20

    cfo = np.asarray(
        [
            row[
                "tail_cfo_loss"
            ]
            for row
            in rows
        ]
    )

    pde = np.asarray(
        [
            row[
                "weighted_pde"
            ]
            for row
            in rows
        ]
    )

    bed = np.asarray(
        [
            row[
                "weighted_bed"
            ]
            for row
            in rows
        ]
    )

    wb = np.asarray(
        [
            row[
                "weighted_wb"
            ]
            for row
            in rows
        ]
    )

    fig = plt.figure(
        figsize=(
            12,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.bar(
        x
        -
        1.5
        *
        width,
        cfo,
        width,
        label=r"$L_{\mathrm{CFO}}$",
    )

    ax.bar(
        x
        -
        0.5
        *
        width,
        pde,
        width,
        label=(
            r"$\lambda_{\mathrm{PDE}}"
            r"L_{\mathrm{PDE}}$"
        ),
    )

    ax.bar(
        x
        +
        0.5
        *
        width,
        bed,
        width,
        label=(
            r"$\lambda_{\mathrm{bed}}"
            r"L_{\mathrm{bed}}$"
        ),
    )

    ax.bar(
        x
        +
        1.5
        *
        width,
        wb,
        width,
        label=(
            r"$\lambda_{\mathrm{WB}}"
            r"L_{\mathrm{WB}}$"
        ),
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Median weighted loss contribution"
    )

    ax.set_title(
        "64x64 loss balance over final training segment"
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
        "02_weighted_loss_balance.png"
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
# ID METRICS
# =====================================================================

def plot_id_metrics(
    rows,
    output_dir,
):

    labels = [
        (
            f"{row['lambda_pde']:g}, "
            f"{row['lambda_bed']:g}, "
            f"{row['lambda_wb']:g}"
        )
        for row
        in rows
    ]

    x = np.arange(
        len(
            rows
        )
    )

    width = 0.26

    validation = np.asarray(
        [
            row[
                "best_validation_l2"
            ]
            for row
            in rows
        ]
    )

    test_l2 = np.asarray(
        [
            row[
                "test_relative_l2"
            ]
            for row
            in rows
        ]
    )

    rmse = np.asarray(
        [
            row[
                "test_rmse"
            ]
            for row
            in rows
        ]
    )

    fig = plt.figure(
        figsize=(
            11,
            6,
        )
    )

    ax = fig.add_subplot(
        111
    )

    ax.bar(
        x
        -
        width,
        validation,
        width,
        label="Best validation Rel-L2",
    )

    ax.bar(
        x,
        test_l2,
        width,
        label="ID test Rel-L2",
    )

    ax.bar(
        x
        +
        width,
        rmse,
        width,
        label="ID test RMSE",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_xlabel(
        r"$(\lambda_{\mathrm{PDE}},"
        r"\lambda_{\mathrm{bed}},"
        r"\lambda_{\mathrm{WB}})$"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "64x64 screening metrics"
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
        "03_id_metrics.png"
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

    results_root = (
        Path(
            args.results_root
        )
        .expanduser()
        .resolve()
    )

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

    metrics_files = sorted(
        results_root
        .rglob(
            "metrics.npz"
        )
    )

    for extra in args.extra_metrics:

        extra_path = (
            Path(extra)
            .expanduser()
            .resolve()
        )

        if extra_path.exists():

            metrics_files.append(
                extra_path
            )

    # Remove duplicates.
    unique = []

    seen = set()

    for path in metrics_files:

        path = path.resolve()

        if path in seen:
            continue

        seen.add(
            path
        )

        unique.append(
            path
        )

    metrics_files = unique

    if not metrics_files:

        raise FileNotFoundError(
            "No metrics.npz files found."
        )

    rows = [
        load_run(
            metrics_path,
            args.tail_fraction,
        )
        for metrics_path
        in metrics_files
    ]

    # Sort by best validation L2 only for convenient viewing.
    # Final scientific choice should also use terrain diagnostics.
    rows.sort(
        key=lambda row:
            (
                row[
                    "best_validation_l2"
                ]
                if np.isfinite(
                    row[
                        "best_validation_l2"
                    ]
                )
                else
                np.inf
            )
    )

    (
        csv_path,
        md_path,
    ) = save_tables(
        rows,
        output_dir,
    )

    validation_plot = (
        plot_validation_history(
            rows,
            output_dir,
        )
    )

    balance_plot = (
        plot_weighted_loss_balance(
            rows,
            output_dir,
        )
    )

    metric_plot = (
        plot_id_metrics(
            rows,
            output_dir,
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "64x64 LAMBDA SWEEP SUMMARY"
    )

    print(
        "=" * 80
    )

    for row in rows:

        print()
        print(
            row[
                "run"
            ]
        )

        print(
            "  lambdas:",
            (
                row[
                    "lambda_pde"
                ],
                row[
                    "lambda_bed"
                ],
                row[
                    "lambda_wb"
                ],
            ),
        )

        print(
            "  best val Rel-L2:",
            f"{row['best_validation_l2']:.6f}",
        )

        print(
            "  test Rel-L2:",
            f"{row['test_relative_l2']:.6f}",
        )

        print(
            "  test RMSE:",
            f"{row['test_rmse']:.6f}",
        )

        print(
            "  weighted PDE / CFO:",
            f"{row['weighted_pde_over_cfo']:.4f}",
        )

        print(
            "  weighted bed / CFO:",
            f"{row['weighted_bed_over_cfo']:.4f}",
        )

        print(
            "  weighted WB / CFO:",
            f"{row['weighted_wb_over_cfo']:.4f}",
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
        "Validation plot:",
        validation_plot,
    )

    print(
        "Loss balance plot:",
        balance_plot,
    )

    print(
        "ID metrics plot:",
        metric_plot,
    )


if __name__ == "__main__":

    main()