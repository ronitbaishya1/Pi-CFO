from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# CONFIGURATION
# ================================================================

ROOT = Path(
    "results/multiseed"
)

OUTPUT_DIR = (
    ROOT
    /
    "summary"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CONDITIONS = {
    "100%": ROOT / "full",
    "25%": ROOT / "time25",
}


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


MODELS = [
    "CFO",
    "PI-CFO",
]


# ================================================================
# READ ONE PHASE-37 CSV
# ================================================================

def read_phase37(
    path: Path,
):

    results = {}

    with path.open(
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            model = row[
                "Model"
            ]

            results[
                model
            ] = {
                metric:
                    float(
                        row[
                            metric
                        ]
                    )
                for metric in METRICS
            }

    return results


# ================================================================
# LOAD ALL SEEDS
# ================================================================

all_results = {}


for condition_name, condition_dir in CONDITIONS.items():

    all_results[
        condition_name
    ] = {
        model: {
            metric: []
            for metric in METRICS
        }
        for model in MODELS
    }

    for seed in [
        0,
        1,
        2,
    ]:

        csv_path = (
            condition_dir
            /
            f"seed{seed}"
            /
            "phase37_results.csv"
        )

        if not csv_path.exists():

            raise FileNotFoundError(
                f"Missing result file:\n{csv_path}"
            )

        seed_results = (
            read_phase37(
                csv_path
            )
        )

        for model in MODELS:

            for metric in METRICS:

                all_results[
                    condition_name
                ][
                    model
                ][
                    metric
                ].append(
                    seed_results[
                        model
                    ][
                        metric
                    ]
                )


# ================================================================
# COMPUTE MEAN ± STANDARD DEVIATION
# ================================================================

summary_rows = []


for condition_name in CONDITIONS:

    for model in MODELS:

        row = {
            "Temporal_data":
                condition_name,

            "Model":
                model,
        }

        for metric in METRICS:

            values = np.asarray(
                all_results[
                    condition_name
                ][
                    model
                ][
                    metric
                ],
                dtype=float,
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            # Sample standard deviation across seeds
            row[
                f"{metric}_std"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        summary_rows.append(
            row
        )


# ================================================================
# SAVE CSV
# ================================================================

csv_columns = [
    "Temporal_data",
    "Model",
]

for metric in METRICS:

    csv_columns.extend(
        [
            f"{metric}_mean",
            f"{metric}_std",
        ]
    )


csv_output = (
    OUTPUT_DIR
    /
    "multiseed_summary.csv"
)


with csv_output.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_columns,
    )

    writer.writeheader()

    writer.writerows(
        summary_rows
    )


# ================================================================
# SAVE MARKDOWN TABLE
# ================================================================

md_output = (
    OUTPUT_DIR
    /
    "multiseed_summary.md"
)


with md_output.open(
    "w"
) as f:

    f.write(
        "# Multi-seed CFO vs PI-CFO Summary\n\n"
    )

    f.write(
        "Mean ± standard deviation across seeds 0, 1, and 2.\n\n"
    )

    f.write(
        "| Temporal Data | Model | "
        "Rel. L2(q) | PDE Residual MSE | Mass Drift |\n"
    )

    f.write(
        "|---|---|---:|---:|---:|\n"
    )

    for row in summary_rows:

        f.write(
            "| "
            f"{row['Temporal_data']} | "
            f"{row['Model']} | "
            f"{row['Rel_L2_q_mean']:.6e} "
            f"± {row['Rel_L2_q_std']:.2e} | "
            f"{row['PDE_residual_MSE_mean']:.6e} "
            f"± {row['PDE_residual_MSE_std']:.2e} | "
            f"{row['Mean_mass_drift_mean']:.6e} "
            f"± {row['Mean_mass_drift_std']:.2e} |\n"
        )


# ================================================================
# PRINT SUMMARY
# ================================================================

print(
    "\n"
    "============================================================"
)

print(
    "MULTI-SEED SUMMARY"
)

print(
    "============================================================"
)


for row in summary_rows:

    print(
        f"\n{row['Temporal_data']} "
        f"{row['Model']}"
    )

    print(
        "  Rel L2(q): "
        f"{row['Rel_L2_q_mean']:.6e} "
        f"± "
        f"{row['Rel_L2_q_std']:.6e}"
    )

    print(
        "  PDE residual: "
        f"{row['PDE_residual_MSE_mean']:.6e} "
        f"± "
        f"{row['PDE_residual_MSE_std']:.6e}"
    )

    print(
        "  Mass drift: "
        f"{row['Mean_mass_drift_mean']:.6e} "
        f"± "
        f"{row['Mean_mass_drift_std']:.6e}"
    )


# ================================================================
# PLOT 1 — RELATIVE L2
# ================================================================

conditions = list(
    CONDITIONS.keys()
)


fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


x = np.arange(
    len(
        conditions
    )
)


for model in MODELS:

    means = []

    stds = []

    for condition in conditions:

        row = next(
            item
            for item in summary_rows
            if (
                item[
                    "Temporal_data"
                ]
                ==
                condition
                and
                item[
                    "Model"
                ]
                ==
                model
            )
        )

        means.append(
            row[
                "Rel_L2_q_mean"
            ]
        )

        stds.append(
            row[
                "Rel_L2_q_std"
            ]
        )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        label=model,
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    conditions
)

ax.set_xlabel(
    "Available temporal observations"
)

ax.set_ylabel(
    "Relative L2(q)"
)

ax.set_title(
    "Multi-seed Prediction Error"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    /
    "multiseed_rel_l2.png",
    dpi=300,
)

plt.close(
    fig
)


# ================================================================
# PLOT 2 — PDE RESIDUAL
# ================================================================

fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


for model in MODELS:

    means = []

    stds = []

    for condition in conditions:

        row = next(
            item
            for item in summary_rows
            if (
                item[
                    "Temporal_data"
                ]
                ==
                condition
                and
                item[
                    "Model"
                ]
                ==
                model
            )
        )

        means.append(
            row[
                "PDE_residual_MSE_mean"
            ]
        )

        stds.append(
            row[
                "PDE_residual_MSE_std"
            ]
        )

    ax.errorbar(
        x,
        means,
        yerr=stds,
        marker="o",
        capsize=5,
        label=model,
    )


ax.set_xticks(
    x
)

ax.set_xticklabels(
    conditions
)

ax.set_xlabel(
    "Available temporal observations"
)

ax.set_ylabel(
    "Mean SWE residual MSE"
)

ax.set_yscale(
    "log"
)

ax.set_title(
    "Multi-seed Physics Residual"
)

ax.grid(
    True,
    alpha=0.3,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    /
    "multiseed_pde_residual.png",
    dpi=300,
)

plt.close(
    fig
)


print(
    "\nOutputs saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCreated:"
)

print(
    "  multiseed_summary.csv"
)

print(
    "  multiseed_summary.md"
)

print(
    "  multiseed_rel_l2.png"
)

print(
    "  multiseed_pde_residual.png"
)