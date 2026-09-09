from __future__ import annotations

"""
Aggregate FINAL FNO CFO / PI-CFO results over seeds 0, 1, and 2
for:

    100% temporal training data
    25% temporal training data

Final PI-CFO physics weight:

    lambda_PDE = 0.1

Expected input structure
------------------------

results/fno_final_lam1e1/

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

results/fno_final_lam1e1/summary/

    all_seed_results.csv

    final_summary.csv
    final_summary.md

    full_summary.csv
    full_summary.md

    time25_summary.csv
    time25_summary.md

    final_rel_l2.png
    final_pde_residual.png
    final_mass_drift.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# PATHS
# ================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "fno_final_lam1e1"
)

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "summary"
)


# ================================================================
# EXPERIMENT SETTINGS
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
# READ ONE PHASE-37 FILE
# ================================================================

def read_phase37_csv(
    path: Path,
) -> list[dict]:
    """Read one Phase-37 CSV."""

    if not path.exists():

        raise FileNotFoundError(
            "\nMissing result file:\n"
            f"{path}\n"
        )

    rows = []

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            if "Model" not in row:

                raise ValueError(
                    f"'Model' column missing from:\n"
                    f"{path}"
                )

            converted = {
                "Model":
                    row["Model"],
            }

            for metric in METRICS:

                if metric not in row:

                    raise ValueError(
                        f"Metric '{metric}' missing from:\n"
                        f"{path}"
                    )

                converted[
                    metric
                ] = float(
                    row[
                        metric
                    ]
                )

            rows.append(
                converted
            )

    return rows


# ================================================================
# LOAD ALL RESULTS
# ================================================================

def load_all_results() -> list[dict]:
    """
    Load:

        2 data regimes
        x 3 seeds
        x 2 models

    = 12 rows.
    """

    all_rows = []

    for regime in REGIMES:

        for seed in SEEDS:

            path = (
                regime[
                    "directory"
                ]
                / f"seed{seed}"
                / "phase37_results.csv"
            )

            rows = read_phase37_csv(
                path
            )

            for model in MODELS:

                matches = [
                    row
                    for row in rows
                    if row[
                        "Model"
                    ] == model
                ]

                if len(
                    matches
                ) != 1:

                    raise ValueError(
                        "\nExpected exactly one result row.\n"
                        f"Temporal data: {regime['label']}\n"
                        f"Seed: {seed}\n"
                        f"Model: {model}\n"
                        f"Found: {len(matches)}\n"
                    )

                result = {
                    "Temporal_Regime":
                        regime[
                            "name"
                        ],

                    "Temporal_Data":
                        regime[
                            "label"
                        ],

                    "Seed":
                        seed,

                    "Model":
                        model,
                }

                for metric in METRICS:

                    result[
                        metric
                    ] = matches[
                        0
                    ][
                        metric
                    ]

                all_rows.append(
                    result
                )

    return all_rows


# ================================================================
# MULTI-SEED STATISTICS
# ================================================================

def build_summary(
    all_rows: list[dict],
) -> list[dict]:
    """Compute mean ± sample standard deviation."""

    summary_rows = []

    for regime in REGIMES:

        for model in MODELS:

            selected = [
                row
                for row in all_rows
                if (
                    row[
                        "Temporal_Regime"
                    ]
                    ==
                    regime[
                        "name"
                    ]
                    and
                    row[
                        "Model"
                    ]
                    ==
                    model
                )
            ]

            if len(
                selected
            ) != len(
                SEEDS
            ):

                raise ValueError(
                    "\nExpected three seeds for:\n"
                    f"{regime['label']} / {model}\n"
                    f"Found: {len(selected)}\n"
                )

            summary = {
                "Temporal_Regime":
                    regime[
                        "name"
                    ],

                "Temporal_Data":
                    regime[
                        "label"
                    ],

                "Model":
                    model,

                "N_seeds":
                    len(
                        selected
                    ),
            }

            for metric in METRICS:

                values = np.asarray(
                    [
                        row[
                            metric
                        ]
                        for row in selected
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
                        values,
                        ddof=1,
                    )
                )

            summary_rows.append(
                summary
            )

    return summary_rows


# ================================================================
# SAVE RAW SEED RESULTS
# ================================================================

def save_all_seed_results(
    rows: list[dict],
) -> None:
    """Save all individual seed results."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_ROOT
        / "all_seed_results.csv"
    )

    columns = [
        "Temporal_Regime",
        "Temporal_Data",
        "Seed",
        "Model",
    ] + METRICS

    with path.open(
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
                    column:
                        row[
                            column
                        ]
                    for column in columns
                }
            )


# ================================================================
# SUMMARY COLUMN LIST
# ================================================================

def summary_columns() -> list[str]:

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
    """Save mean/std summary."""

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_ROOT
        / filename
    )

    columns = summary_columns()

    with path.open(
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
                    column:
                        row[
                            column
                        ]
                    for column in columns
                }
            )


# ================================================================
# SAVE MARKDOWN
# ================================================================

def save_summary_markdown(
    rows: list[dict],
    filename: str,
    title: str,
) -> None:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_ROOT
        / filename
    )

    with path.open(
        "w"
    ) as f:

        f.write(
            f"# {title}\n\n"
        )

        f.write(
            "Architecture: FNO2d\n\n"
        )

        f.write(
            "PI-CFO lambda_PDE: 0.1\n\n"
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
    rows: list[dict],
    regime_name: str,
    model: str,
) -> dict:

    matches = [
        row
        for row in rows
        if (
            row[
                "Temporal_Regime"
            ]
            ==
            regime_name
            and
            row[
                "Model"
            ]
            ==
            model
        )
    ]

    if len(
        matches
    ) != 1:

        raise ValueError(
            f"Could not uniquely find "
            f"{regime_name} / {model}."
        )

    return matches[
        0
    ]


# ================================================================
# PLOT METRIC
# ================================================================

def plot_metric(
    summary_rows,
    metric,
    ylabel,
    title,
    filename,
):

    x = np.arange(
        2
    )

    width = 0.35

    cfo_means = []
    cfo_stds = []

    picfo_means = []
    picfo_stds = []

    for regime_name in [
        "full",
        "time25",
    ]:

        cfo = get_summary_row(
            summary_rows,
            regime_name,
            "CFO",
        )

        picfo = get_summary_row(
            summary_rows,
            regime_name,
            "PI-CFO",
        )

        cfo_means.append(
            cfo[
                f"{metric}_mean"
            ]
        )

        cfo_stds.append(
            cfo[
                f"{metric}_std"
            ]
        )

        picfo_means.append(
            picfo[
                f"{metric}_mean"
            ]
        )

        picfo_stds.append(
            picfo[
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
        [
            "100%",
            "25%",
        ]
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

    plt.close(
        fig
    )


# ================================================================
# PRINT SUMMARY
# ================================================================

def print_summary(
    summary_rows,
):

    print(
        "\n"
        "============================================================"
    )

    print(
        "FINAL FNO MULTI-SEED SUMMARY"
    )

    print(
        "PI-CFO lambda_PDE = 0.1"
    )

    print(
        "============================================================"
    )

    for regime in REGIMES:

        print(
            "\n"
            f"TEMPORAL DATA: "
            f"{regime['label']}"
        )

        print(
            "=" * 60
        )

        for model in MODELS:

            row = get_summary_row(
                summary_rows,
                regime[
                    "name"
                ],
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


# ================================================================
# MAIN
# ================================================================

def main():

    print(
        "\nChecking final result files..."
    )

    for regime in REGIMES:

        print(
            "\n"
            f"{regime['label']} temporal data"
        )

        for seed in SEEDS:

            path = (
                regime[
                    "directory"
                ]
                / f"seed{seed}"
                / "phase37_results.csv"
            )

            print(
                f"  Seed {seed}: {path}"
            )

            if not path.exists():

                raise FileNotFoundError(
                    "\nRequired final result file missing:\n"
                    f"{path}\n"
                )

    all_rows = load_all_results()

    summary_rows = build_summary(
        all_rows
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_all_seed_results(
        all_rows
    )

    # ------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------

    save_summary_csv(
        summary_rows,
        "final_summary.csv",
    )

    save_summary_markdown(
        summary_rows,
        "final_summary.md",
        (
            "Final FNO Multi-Seed Summary — "
            "100% vs 25% Temporal Data"
        ),
    )

    # ------------------------------------------------------------
    # Full only
    # ------------------------------------------------------------

    full_rows = [
        row
        for row in summary_rows
        if row[
            "Temporal_Regime"
        ] == "full"
    ]

    save_summary_csv(
        full_rows,
        "full_summary.csv",
    )

    save_summary_markdown(
        full_rows,
        "full_summary.md",
        "Final FNO Summary — 100% Temporal Data",
    )

    # ------------------------------------------------------------
    # 25% only
    # ------------------------------------------------------------

    time25_rows = [
        row
        for row in summary_rows
        if row[
            "Temporal_Regime"
        ] == "time25"
    ]

    save_summary_csv(
        time25_rows,
        "time25_summary.csv",
    )

    save_summary_markdown(
        time25_rows,
        "time25_summary.md",
        "Final FNO Summary — 25% Temporal Data",
    )

    # ------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------

    plot_metric(
        summary_rows,
        metric="Rel_L2_q",
        ylabel="Relative L2 Error",
        title="Final FNO Prediction Error",
        filename="final_rel_l2.png",
    )

    plot_metric(
        summary_rows,
        metric="PDE_residual_MSE",
        ylabel="Mean SWE Residual MSE",
        title="Final FNO SWE Residual",
        filename="final_pde_residual.png",
    )

    plot_metric(
        summary_rows,
        metric="Mean_mass_drift",
        ylabel="Mean Relative Mass Drift",
        title="Final FNO Mass Drift",
        filename="final_mass_drift.png",
    )

    print_summary(
        summary_rows
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "FINAL AGGREGATION COMPLETE"
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


if __name__ == "__main__":
    main()