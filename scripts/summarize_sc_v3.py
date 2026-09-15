from pathlib import Path

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RESULTS_ROOT = (
    PROJECT_ROOT
    /
    "results"
    /
    "sc_v3_bathy_bed_pi"
)


def load_scalar(data, name):

    if name not in data:

        return float("nan")

    return float(
        np.asarray(
            data[name]
        ).reshape(
            -1
        )[0]
    )


def main():

    print()
    print("=" * 100)
    print("SC-FNO-v3 TRAIN-AND-UNROLL SUMMARY")
    print("=" * 100)

    print(
        f"{'K':>4}"
        f"{'Best epoch':>14}"
        f"{'Best val L2':>18}"
        f"{'ID test L2':>18}"
        f"{'RMSE':>14}"
        f"{'Rel Fro':>14}"
    )

    print("-" * 100)

    rows = []

    for depth in [
        1,
        2,
        3,
        4,
    ]:

        path = (
            RESULTS_ROOT
            /
            f"depth{depth}"
            /
            "seed0"
            /
            "metrics.npz"
        )

        if not path.exists():

            print(
                f"{depth:>4}"
                f"{'MISSING':>14}"
            )

            continue

        data = np.load(
            path
        )

        best_epoch = int(
            load_scalar(
                data,
                "best_epoch",
            )
        )

        best_val = load_scalar(
            data,
            "best_eval_relative_l2",
        )

        test_l2 = load_scalar(
            data,
            "test_relative_l2",
        )

        test_rmse = load_scalar(
            data,
            "test_rmse",
        )

        test_fro = load_scalar(
            data,
            "test_relative_frobenius",
        )

        rows.append(
            {
                "depth":
                    depth,

                "best_epoch":
                    best_epoch,

                "best_val":
                    best_val,

                "test_l2":
                    test_l2,

                "rmse":
                    test_rmse,

                "fro":
                    test_fro,
            }
        )

        print(
            f"{depth:>4}"
            f"{best_epoch:>14d}"
            f"{best_val:>18.6f}"
            f"{test_l2:>18.6f}"
            f"{test_rmse:>14.6f}"
            f"{test_fro:>14.6f}"
        )

    print("=" * 100)

    if not rows:

        print(
            "\nNo completed stages were found."
        )

        return

    best_by_validation = min(
        rows,
        key=lambda row:
        row[
            "best_val"
        ],
    )

    print()
    print(
        "BEST DEPTH BY VALIDATION:"
    )

    print(
        f"K = "
        f"{best_by_validation['depth']}"
    )

    print(
        f"Validation Rel L2 = "
        f"{best_by_validation['best_val']:.6f}"
    )

    print(
        f"ID Test Rel L2 = "
        f"{best_by_validation['test_l2']:.6f}"
    )

    print()

    ordinary_fno_reference = (
        0.01544
    )

    difference = (
        best_by_validation[
            "best_val"
        ]
        -
        ordinary_fno_reference
    )

    relative_difference = (
        difference
        /
        ordinary_fno_reference
    )

    print(
        "REFERENCE ORDINARY FNO "
        "BED-PI-CFO VALIDATION L2:"
    )

    print(
        f"{ordinary_fno_reference:.6f}"
    )

    print()

    print(
        "SC-v3 relative difference "
        "from ordinary FNO:"
    )

    print(
        f"{100.0 * relative_difference:+.2f}%"
    )

    print()

    if (
        best_by_validation[
            "best_val"
        ]
        <=
        0.018
    ):

        print(
            "DECISION: SC-v3 is competitive "
            "enough for OOD bathymetry testing."
        )

    elif (
        best_by_validation[
            "best_val"
        ]
        <=
        0.020
    ):

        print(
            "DECISION: borderline."
        )

        print(
            "OOD testing may still be useful "
            "because SC may generalize better "
            "despite slightly worse ID error."
        )

    else:

        print(
            "DECISION: SC-v3 remains clearly "
            "behind the ordinary FNO baseline."
        )

        print(
            "Do not add well-balanced loss "
            "to SC-v3 yet."
        )


if __name__ == "__main__":

    main()