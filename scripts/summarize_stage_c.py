from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(
    "results/stage_c"
)

VARIANTS = [
    "baseline",
    "tan001",
    "tan005",
    "tan01",
]

LABELS = {
    "baseline":
        "Baseline",

    "tan001":
        "lambda_0.001",

    "tan005":
        "lambda_0.005",

    "tan01":
        "lambda_0.01",
}


def scalar(
    archive,
    key,
):

    return float(
        np.asarray(
            archive[key]
        )
    )


def load_metric(
    scenario,
    variant,
):

    path = (
        ROOT
        /
        scenario
        /
        variant
        /
        "metrics.npz"
    )

    return np.load(path)


def main():

    rows = []

    for variant in VARIANTS:

        id32 = load_metric(
            "id32",
            variant,
        )

        ood32 = load_metric(
            "ood32",
            variant,
        )

        perturbed = load_metric(
            "perturbed32",
            variant,
        )

        long_run = load_metric(
            "long32_t2",
            variant,
        )

        cross64 = load_metric(
            "cross64",
            variant,
        )

        cross64_ood = load_metric(
            "cross64_ood",
            variant,
        )

        row = {

            "model":
                LABELS[
                    variant
                ],

            "id32_rel_l2":
                scalar(
                    id32,
                    "global_relative_l2",
                ),

            "ood32_rel_l2":
                scalar(
                    ood32,
                    "global_relative_l2",
                ),

            "perturbed32_rel_l2":
                scalar(
                    perturbed,
                    "global_relative_l2",
                ),

            "long_t2_global_rel_l2":
                scalar(
                    long_run,
                    "global_relative_l2",
                ),

            "long_t2_final_rel_l2":
                float(
                    long_run[
                        "error_by_time"
                    ][-1]
                ),

            "cross64_rel_l2":
                scalar(
                    cross64,
                    "global_relative_l2",
                ),

            "cross64_ood_rel_l2":
                scalar(
                    cross64_ood,
                    "global_relative_l2",
                ),

            "cross64_eta_rel_l2":
                scalar(
                    cross64,
                    "eta_relative_l2",
                ),

            "long_t2_final_mass_error":
                float(
                    long_run[
                        "mass_relative_error"
                    ][-1]
                ),
        }

        rows.append(row)

    output = (
        ROOT
        /
        "stage_c_summary.csv"
    )

    with open(
        output,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 120)
    print("FINAL STAGE-C SUMMARY")
    print("=" * 120)

    header = (
        f"{'model':<16}"
        f"{'ID32':>12}"
        f"{'OOD32':>12}"
        f"{'perturb':>12}"
        f"{'long-final':>14}"
        f"{'32->64':>12}"
        f"{'64-OOD':>12}"
    )

    print(header)

    for row in rows:

        print(
            f"{row['model']:<16}"
            f"{row['id32_rel_l2']:>12.6f}"
            f"{row['ood32_rel_l2']:>12.6f}"
            f"{row['perturbed32_rel_l2']:>12.6f}"
            f"{row['long_t2_final_rel_l2']:>14.6f}"
            f"{row['cross64_rel_l2']:>12.6f}"
            f"{row['cross64_ood_rel_l2']:>12.6f}"
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()