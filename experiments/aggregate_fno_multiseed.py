from __future__ import annotations

"""
Aggregate FNO CFO / PI-CFO results over seeds 0, 1, and 2
for both 100% and 25% temporal training data.

Expected inputs
---------------

results/fno_multiseed/
    full/
        seed0/phase37_results.csv
        seed1/phase37_results.csv
        seed2/phase37_results.csv

    time25/
        seed0/phase37_results.csv
        seed1/phase37_results.csv
        seed2/phase37_results.csv


Outputs
-------

results/fno_multiseed/summary/
    all_seed_results.csv

    summary.csv
    summary.md

    full_summary.csv
    full_summary.md

    time25_summary.csv
    time25_summary.md

    multiseed_rel_l2.png
    multiseed_pde_residual.png
    multiseed_mass_drift.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fno_multiseed"
)

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "summary"
)


# ================================================================
# SETTINGS
# ================================================================

SEEDS = [
    0,
    1,
    2,
]

MODELS = [
    "CFO",
    "PI-CFO",
]

REGIMES = [
    {
        "name": "full",
        "label": "100%",
        "directory": RESULTS_ROOT / "full",
    },
    {
        "name": "time25",
        "label": "25%",
        "directory": RESULTS_ROOT / "time25",
    },
]

METRICS = [
    "Rel_L2_q",
    "E_h",
    "E_hu",
    "E_hv",
    "PDE_residual_MSE",
    "Mean_mass_drift",
    "Final_mass_drift",
    "RMSE",
    "Relative_Frobenius",
]


# ================================================================
# READ ONE PHASE-37 CSV
# ================================================================

def read_phase37_csv(
    csv_path: Path,
) -> list[dict]:
    """Read a single phase37_results.csv."""

    if not csv_path.exists():
        raise FileNotFoundError(
            "\nMissing Phase-37 result file:\n"
            f"{csv_path}\n"
        )

    rows = []

    with csv_path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if "Model" not in row:
                raise ValueError(
                    f"'Model' column missing from:\n"
                    f"{csv_path}"
                )

            converted = {
                "Model": row["Model"],
            }

            for metric in METRICS:

                if metric not in row:
                    raise ValueError(
                        f"Metric '{metric}' missing from:\n"
                        f"{csv_path}"
                    )

                converted[metric] = float(
                    row[metric]
                )

            rows.append(
                converted
            )

    return rows


# ================================================================
# LOAD ALL 12 MODEL RESULTS
# ================================================================

def load_all_results() -> list[dict]:
    """
    Load:

        2 temporal-data regimes
        x 3 seeds
        x 2 models

    = 12 model result rows.
    """

    all_rows = []

    for regime in REGIMES:

        for seed in SEEDS:

            csv_path = (
                regime["directory"]
                / f"seed{seed}"
                / "phase37_results.csv"
            )

            rows = read_phase37_csv(
                csv_path
            )

            for model in MODELS:

                matches = [
                    row
                    for row in rows
                    if row["Model"] == model
                ]

                if len(matches) != 1:
                    raise ValueError(
                        "\nExpected exactly one result row for:\n"
                        f"Temporal data: {regime['label']}\n"
                        f"Seed: {seed}\n"
                        f"Model: {model}\n"
                        f"Found: {len(matches)}\n"
                        f"File: {csv_path}\n"
                    )

                result = {
                    "Temporal_Regime":
                        regime["name"],

                    "Temporal_Data":
                        regime["label"],

                    "Seed":
                        seed,

                    "Model":
                        model,
                }

                for metric in METRICS:
                    result[metric] = (
                        matches[0][metric]
                    )

                all_rows.append(
                    result
                )

    return all_rows


# ================================================================
# BUILD MULTI-SEED SUMMARY
# ================================================================

def build_summary(
    all_rows: list[dict],
) -> list[dict]:
    """
    Compute mean and sample standard deviation
    across seeds 0, 1, and 2.

    Standard deviation uses ddof=1.
    """

    summary_rows = []

    for regime in REGIMES:

        for model in MODELS:

            selected = [
                row
                for row in all_rows
                if (
                    row["Temporal_Regime"]
                    == regime["name"]
                    and
                    row["Model"]
                    == model
                )
            ]

            if len(selected) != len(SEEDS):
                raise ValueError(
                    "\nExpected three seeds for:\n"
                    f"{regime['label']} / {model}\n"
                    f"Found: {len(selected)}\n"
                )

            summary = {
                "Temporal_Regime":
                    regime["name"],

                "Temporal_Data":
                    regime["label"],

                "Model":
                    model,

                "N_seeds":
                    len(selected),
            }

            for metric in METRICS:

                values = np.asarray(
                    [
                        row[metric]
                        for row in selected
                    ],
                    dtype=np.float64,
                )

                summary[
                    f"{metric}_mean"
                ] = float(
                    np.mean(values)
                )

                summary[
                    f"{metric}_std"
                ] = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

            summary_rows.append(
                summary
            )

    return summary_rows


# ================================================================
# SAVE ALL INDIVIDUAL SEED RESULTS
# ================================================================

def save_all_seed_results(
    all_rows: list[dict],
) -> None:
    """Save all 12 individual result rows."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "all_seed_results.csv"
    )

    columns = [
        "Temporal_Regime",
        "Temporal_Data",
        "Seed",
        "Model",
    ] + METRICS

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in all_rows:

            writer.writerow(
                {
                    column: row[column]
                    for column in columns
                }
            )


# ================================================================
# SUMMARY COLUMN NAMES
# ================================================================

def summary_columns() -> list[str]:
    """Return column names for summary CSV files."""

    columns = [
        "Temporal_Regime",
        "Temporal_Data",
        "Model",
        "N_seeds",
    ]

    for metric in METRICS:

        columns.append(
            f"{metric}_mean"
        )

        columns.append(
            f"{metric}_std"
        )

    return columns


# ================================================================
# SAVE SUMMARY CSV
# ================================================================

def save_summary_csv(
    rows: list[dict],
    filename: str,
) -> None:
    """Save aggregate results as CSV."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / filename
    )

    columns = summary_columns()

    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    column: row[column]
                    for column in columns
                }
            )


# ================================================================
# SAVE SUMMARY MARKDOWN
# ================================================================

def save_summary_markdown(
    rows: list[dict],
    filename: str,
    title: str,
) -> None:
    """Save human-readable mean ± std table."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / filename
    )

    with output_path.open(
        "w"
    ) as f:

        f.write(
            f"# {title}\n\n"
        )

        f.write(
            "Seeds: 0, 1, 2\n\n"
        )

        f.write(
            "Values are mean ± sample standard deviation.\n\n"
        )

        f.write(
            "| Temporal Data | Model | Rel L2(q) | "
            "E_h | E_hu | E_hv | PDE Residual MSE | "
            "Mean Mass Drift | Final Mass Drift | RMSE |\n"
        )

        f.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

        for row in rows:

            f.write(
                "| "
                f"{row['Temporal_Data']} | "
                f"{row['Model']} | "

                f"{row['Rel_L2_q_mean']:.6e} ± "
                f"{row['Rel_L2_q_std']:.2e} | "

                f"{row['E_h_mean']:.6e} ± "
                f"{row['E_h_std']:.2e} | "

                f"{row['E_hu_mean']:.6e} ± "
                f"{row['E_hu_std']:.2e} | "

                f"{row['E_hv_mean']:.6e} ± "
                f"{row['E_hv_std']:.2e} | "

                f"{row['PDE_residual_MSE_mean']:.6e} ± "
                f"{row['PDE_residual_MSE_std']:.2e} | "

                f"{row['Mean_mass_drift_mean']:.6e} ± "
                f"{row['Mean_mass_drift_std']:.2e} | "

                f"{row['Final_mass_drift_mean']:.6e} ± "
                f"{row['Final_mass_drift_std']:.2e} | "

                f"{row['RMSE_mean']:.6e} ± "
                f"{row['RMSE_std']:.2e} |\n"
            )


# ================================================================
# FIND ONE SUMMARY ROW
# ================================================================

def get_summary_row(
    summary_rows: list[dict],
    regime_name: str,
    model: str,
) -> dict:
    """Return one requested aggregate row."""

    matches = [
        row
        for row in summary_rows
        if (
            row["Temporal_Regime"]
            == regime_name
            and
            row["Model"]
            == model
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Could not uniquely find summary row for "
            f"{regime_name} / {model}."
        )

    return matches[0]


# ================================================================
# PLOT ONE METRIC
# ================================================================

def plot_metric(
    summary_rows: list[dict],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """Plot CFO vs PI-CFO for 100% and 25% data."""

    regime_names = [
        "full",
        "time25",
    ]

    regime_labels = [
        "100%",
        "25%",
    ]

    x = np.arange(
        len(regime_names)
    )

    width = 0.35

    cfo_means = []
    cfo_stds = []

    picfo_means = []
    picfo_stds = []

    for regime_name in regime_names:

        cfo_row = get_summary_row(
            summary_rows,
            regime_name,
            "CFO",
        )

        picfo_row = get_summary_row(
            summary_rows,
            regime_name,
            "PI-CFO",
        )

        cfo_means.append(
            cfo_row[
                f"{metric}_mean"
            ]
        )

        cfo_stds.append(
            cfo_row[
                f"{metric}_std"
            ]
        )

        picfo_means.append(
            picfo_row[
                f"{metric}_mean"
            ]
        )

        picfo_stds.append(
            picfo_row[
                f"{metric}_std"
            ]
        )

    fig, ax = plt.subplots(
        figsize=(
            7.0,
            5.0,
        )
    )

    ax.bar(
        x - width / 2,
        cfo_means,
        width,
        yerr=cfo_stds,
        capsize=5,
        label="CFO",
    )

    ax.bar(
        x + width / 2,
        picfo_means,
        width,
        yerr=picfo_stds,
        capsize=5,
        label="PI-CFO",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regime_labels
    )

    ax.set_xlabel(
        "Temporal training data retained"
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

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_ROOT
        / filename,
        dpi=300,
    )

    plt.close(fig)


# ================================================================
# PRINT INDIVIDUAL SEED RESULTS
# ================================================================

def print_seed_results(
    all_rows: list[dict],
) -> None:
    """Print all individual seed-level results."""

    print(
        "\n"
        "============================================================"
    )

    print(
        "FNO SEED-LEVEL RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"{'Data':<8}"
        f"{'Seed':<8}"
        f"{'Model':<12}"
        f"{'Rel L2(q)':>16}"
        f"{'PDE MSE':>16}"
        f"{'Mass drift':>16}"
    )

    print(
        "-" * 76
    )

    for row in all_rows:

        print(
            f"{row['Temporal_Data']:<8}"
            f"{row['Seed']:<8}"
            f"{row['Model']:<12}"
            f"{row['Rel_L2_q']:>16.6e}"
            f"{row['PDE_residual_MSE']:>16.6e}"
            f"{row['Mean_mass_drift']:>16.6e}"
        )


# ================================================================
# PRINT AGGREGATE SUMMARY
# ================================================================

def print_summary(
    summary_rows: list[dict],
) -> None:
    """Print final mean ± std values."""

    print(
        "\n"
        "============================================================"
    )

    print(
        "FNO MULTI-SEED SUMMARY"
    )

    print(
        "100% AND 25% TEMPORAL DATA"
    )

    print(
        "============================================================"
    )

    for regime in REGIMES:

        print(
            "\n"
            f"TEMPORAL DATA: {regime['label']}"
        )

        print(
            "=" * 58
        )

        for model in MODELS:

            row = get_summary_row(
                summary_rows,
                regime["name"],
                model,
            )

            print(
                f"\n{model}"
            )

            print(
                "-" * 45
            )

            print(
                "Rel L2(q): "
                f"{row['Rel_L2_q_mean']:.8e} "
                f"± {row['Rel_L2_q_std']:.8e}"
            )

            print(
                "E_h: "
                f"{row['E_h_mean']:.8e} "
                f"± {row['E_h_std']:.8e}"
            )

            print(
                "E_hu: "
                f"{row['E_hu_mean']:.8e} "
                f"± {row['E_hu_std']:.8e}"
            )

            print(
                "E_hv: "
                f"{row['E_hv_mean']:.8e} "
                f"± {row['E_hv_std']:.8e}"
            )

            print(
                "PDE residual MSE: "
                f"{row['PDE_residual_MSE_mean']:.8e} "
                f"± {row['PDE_residual_MSE_std']:.8e}"
            )

            print(
                "Mean mass drift: "
                f"{row['Mean_mass_drift_mean']:.8e} "
                f"± {row['Mean_mass_drift_std']:.8e}"
            )

            print(
                "Final mass drift: "
                f"{row['Final_mass_drift_mean']:.8e} "
                f"± {row['Final_mass_drift_std']:.8e}"
            )

            print(
                "RMSE: "
                f"{row['RMSE_mean']:.8e} "
                f"± {row['RMSE_std']:.8e}"
            )

            print(
                "Relative Frobenius: "
                f"{row['Relative_Frobenius_mean']:.8e} "
                f"± {row['Relative_Frobenius_std']:.8e}"
            )


# ================================================================
# MAIN
# ================================================================

def main() -> None:
    """Run combined 100% + 25% FNO multi-seed aggregation."""

    print(
        "\nChecking all required result files..."
    )

    for regime in REGIMES:

        print(
            "\n"
            f"{regime['label']} temporal data:"
        )

        for seed in SEEDS:

            csv_path = (
                regime["directory"]
                / f"seed{seed}"
                / "phase37_results.csv"
            )

            print(
                f"  Seed {seed}: {csv_path}"
            )

            if not csv_path.exists():
                raise FileNotFoundError(
                    "\nRequired result file is missing:\n"
                    f"{csv_path}\n"
                )

    all_rows = load_all_results()

    summary_rows = build_summary(
        all_rows
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # All individual results
    # ------------------------------------------------------------

    save_all_seed_results(
        all_rows
    )

    # ------------------------------------------------------------
    # Combined 100% + 25% summary
    # ------------------------------------------------------------

    save_summary_csv(
        summary_rows,
        "summary.csv",
    )

    save_summary_markdown(
        summary_rows,
        "summary.md",
        "FNO Multi-Seed Summary — 100% vs 25% Temporal Data",
    )

    # ------------------------------------------------------------
    # 100% only
    # ------------------------------------------------------------

    full_rows = [
        row
        for row in summary_rows
        if row["Temporal_Regime"] == "full"
    ]

    save_summary_csv(
        full_rows,
        "full_summary.csv",
    )

    save_summary_markdown(
        full_rows,
        "full_summary.md",
        "FNO Multi-Seed Summary — 100% Temporal Data",
    )

    # ------------------------------------------------------------
    # 25% only
    # ------------------------------------------------------------

    time25_rows = [
        row
        for row in summary_rows
        if row["Temporal_Regime"] == "time25"
    ]

    save_summary_csv(
        time25_rows,
        "time25_summary.csv",
    )

    save_summary_markdown(
        time25_rows,
        "time25_summary.md",
        "FNO Multi-Seed Summary — 25% Temporal Data",
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_metric(
        summary_rows,
        metric="Rel_L2_q",
        ylabel="Relative L2 Error",
        title="FNO: Relative L2 Error",
        filename="multiseed_rel_l2.png",
    )

    plot_metric(
        summary_rows,
        metric="PDE_residual_MSE",
        ylabel="Mean SWE Residual MSE",
        title="FNO: SWE Residual",
        filename="multiseed_pde_residual.png",
    )

    plot_metric(
        summary_rows,
        metric="Mean_mass_drift",
        ylabel="Mean Relative Mass Drift",
        title="FNO: Mass Drift",
        filename="multiseed_mass_drift.png",
    )

    # ------------------------------------------------------------
    # Terminal output
    # ------------------------------------------------------------

    print_seed_results(
        all_rows
    )

    print_summary(
        summary_rows
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "AGGREGATION COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_ROOT
    )

    print(
        "\nCreated files:"
    )

    print(
        "  all_seed_results.csv"
    )

    print(
        "  summary.csv"
    )

    print(
        "  summary.md"
    )

    print(
        "  full_summary.csv"
    )

    print(
        "  full_summary.md"
    )

    print(
        "  time25_summary.csv"
    )

    print(
        "  time25_summary.md"
    )

    print(
        "  multiseed_rel_l2.png"
    )

    print(
        "  multiseed_pde_residual.png"
    )

    print(
        "  multiseed_mass_drift.png"
    )


if __name__ == "__main__":
    main()