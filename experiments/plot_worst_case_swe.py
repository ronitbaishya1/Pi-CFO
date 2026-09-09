from __future__ import annotations

"""
Create CFO-paper-style worst-case prediction plots for the
2D shallow-water equation.

State layout
------------

q = [h, hu, hv]

Saved array shape
-----------------

(batch, time, nx, ny, 3)

The script:

1. Loads ground truth, CFO and PI-CFO predictions.
2. Finds the worst test trajectory using full-trajectory relative L2 error.
3. Extracts a horizontal centerline through the 2D domain.
4. Plots Ground Truth, CFO and PI-CFO at four physical times.
5. Creates separate figures for h, hu and hv.

Example output
--------------

worst_case_h_centerline.png
worst_case_hu_centerline.png
worst_case_hv_centerline.png

A text summary is also saved containing the selected trajectory index.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# CONSTANTS
# ================================================================

EPS = 1e-12

CHANNEL_NAMES = [
    "h",
    "hu",
    "hv",
]

CHANNEL_LABELS = [
    r"$h(x,y,t)$",
    r"$hu(x,y,t)$",
    r"$hv(x,y,t)$",
]


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create CFO-style worst-case centerline plots "
            "for shallow-water predictions."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help=(
            "Directory containing test_ground_truth.npy, "
            "cfo_predictions.npy and picfo_predictions.npy."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[
            0.25,
            0.50,
            0.75,
            1.00,
        ],
        help="Physical times to show.",
    )

    parser.add_argument(
        "--domain-min",
        type=float,
        default=-2.5,
    )

    parser.add_argument(
        "--domain-max",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--worst-model",
        type=str,
        default="picfo",
        choices=[
            "cfo",
            "picfo",
            "max",
        ],
        help=(
            "How to select the worst trajectory. "
            "'cfo' uses CFO error, "
            "'picfo' uses PI-CFO error, "
            "'max' uses the larger error of the two."
        ),
    )

    return parser.parse_args()


# ================================================================
# RELATIVE L2 ERROR
# ================================================================

def trajectory_relative_l2(
    target: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    """
    Compute one relative L2 error per test trajectory.

    Parameters
    ----------
    target:
        Shape (batch, time, nx, ny, channels)

    prediction:
        Same shape.

    Returns
    -------
    errors:
        Shape (batch,)
    """

    difference = (
        prediction
        -
        target
    )

    numerator = np.sqrt(
        np.sum(
            difference**2,
            axis=(
                1,
                2,
                3,
                4,
            ),
        )
    )

    denominator = np.sqrt(
        np.sum(
            target**2,
            axis=(
                1,
                2,
                3,
                4,
            ),
        )
    )

    return (
        numerator
        /
        (
            denominator
            +
            EPS
        )
    )


# ================================================================
# SELECT WORST TRAJECTORY
# ================================================================

def select_worst_trajectory(
    *,
    target,
    cfo_pred,
    picfo_pred,
    selection_mode,
):
    """Select worst test trajectory."""

    cfo_errors = trajectory_relative_l2(
        target,
        cfo_pred,
    )

    picfo_errors = trajectory_relative_l2(
        target,
        picfo_pred,
    )

    if selection_mode == "cfo":

        selection_error = (
            cfo_errors
        )

    elif selection_mode == "picfo":

        selection_error = (
            picfo_errors
        )

    elif selection_mode == "max":

        selection_error = np.maximum(
            cfo_errors,
            picfo_errors,
        )

    else:

        raise ValueError(
            f"Unknown selection mode: "
            f"{selection_mode}"
        )

    worst_index = int(
        np.argmax(
            selection_error
        )
    )

    return {
        "index":
            worst_index,

        "cfo_errors":
            cfo_errors,

        "picfo_errors":
            picfo_errors,

        "selection_error":
            selection_error,
    }


# ================================================================
# TIME INDICES
# ================================================================

def nearest_time_indices(
    times: np.ndarray,
    requested_times,
):
    """Find nearest saved index for each requested physical time."""

    indices = []

    for requested in requested_times:

        index = int(
            np.argmin(
                np.abs(
                    times
                    -
                    requested
                )
            )
        )

        indices.append(
            index
        )

    return indices


# ================================================================
# CENTERLINE EXTRACTION
# ================================================================

def extract_horizontal_centerline(
    field: np.ndarray,
):
    """
    Extract horizontal centerline.

    field shape:
        (nx, ny)

    Returns:
        field[:, ny//2]
    """

    ny = field.shape[1]

    y_index = (
        ny
        //
        2
    )

    return (
        field[
            :,
            y_index,
        ],
        y_index,
    )


# ================================================================
# PLOT ONE CHANNEL
# ================================================================

def plot_channel(
    *,
    target,
    cfo_pred,
    picfo_pred,
    trajectory_index,
    channel_index,
    physical_times,
    time_indices,
    x_coordinates,
    output_dir,
):
    """Create CFO-paper-style four-panel line plot."""

    n_panels = len(
        time_indices
    )

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            4.0 * n_panels,
            3.6,
        ),
        squeeze=False,
        sharey=True,
    )

    axes = axes[0]

    y_index_used = None

    for panel_index, time_index in enumerate(
        time_indices
    ):

        ax = axes[
            panel_index
        ]

        truth_field = target[
            trajectory_index,
            time_index,
            ...,
            channel_index,
        ]

        cfo_field = cfo_pred[
            trajectory_index,
            time_index,
            ...,
            channel_index,
        ]

        picfo_field = picfo_pred[
            trajectory_index,
            time_index,
            ...,
            channel_index,
        ]

        (
            truth_line,
            y_index,
        ) = extract_horizontal_centerline(
            truth_field
        )

        (
            cfo_line,
            _,
        ) = extract_horizontal_centerline(
            cfo_field
        )

        (
            picfo_line,
            _,
        ) = extract_horizontal_centerline(
            picfo_field
        )

        y_index_used = (
            y_index
        )

        # --------------------------------------------------------
        # Plot
        # --------------------------------------------------------

        ax.plot(
            x_coordinates,
            truth_line,
            label="Ref.",
            linewidth=2.0,
        )

        ax.plot(
            x_coordinates,
            cfo_line,
            label="CFO Pred.",
            linewidth=2.0,
            linestyle="--",
        )

        ax.plot(
            x_coordinates,
            picfo_line,
            label="PI-CFO Pred.",
            linewidth=2.0,
            linestyle=":",
        )

        # --------------------------------------------------------
        # Title
        # --------------------------------------------------------

        ax.set_title(
            (
                f"t = "
                f"{physical_times[panel_index]:.2f}"
            ),
            fontsize=12,
        )

        ax.set_xlabel(
            "x"
        )

        ax.grid(
            True,
            alpha=0.2,
        )

        if panel_index == 0:

            ax.set_ylabel(
                CHANNEL_LABELS[
                    channel_index
                ]
            )

            ax.legend(
                fontsize=9,
            )

    # ============================================================
    # Figure title
    # ============================================================

    channel_name = (
        CHANNEL_NAMES[
            channel_index
        ]
    )

    fig.suptitle(
        (
            f"Worst-Case SWE Prediction: "
            f"{channel_name} Centerline\n"
            f"Test trajectory {trajectory_index}"
        ),
        fontsize=14,
    )

    fig.tight_layout()

    output_path = (
        output_dir
        /
        (
            f"worst_case_"
            f"{channel_name}_"
            f"centerline.png"
        )
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return y_index_used


# ================================================================
# SAVE SUMMARY
# ================================================================

def save_summary(
    *,
    output_dir,
    worst_info,
    physical_times,
    time_indices,
    y_index,
):
    """Save selected trajectory and errors."""

    path = (
        output_dir
        /
        "worst_case_summary.txt"
    )

    index = (
        worst_info[
            "index"
        ]
    )

    cfo_error = (
        worst_info[
            "cfo_errors"
        ][
            index
        ]
    )

    picfo_error = (
        worst_info[
            "picfo_errors"
        ][
            index
        ]
    )

    with path.open(
        "w"
    ) as f:

        f.write(
            "Worst-Case SWE Centerline Plot\n"
        )

        f.write(
            "==============================\n\n"
        )

        f.write(
            f"Selected test trajectory: "
            f"{index}\n"
        )

        f.write(
            f"CFO trajectory relative L2: "
            f"{cfo_error:.8e}\n"
        )

        f.write(
            f"PI-CFO trajectory relative L2: "
            f"{picfo_error:.8e}\n"
        )

        f.write(
            f"Horizontal centerline y-index: "
            f"{y_index}\n\n"
        )

        f.write(
            "Displayed times:\n"
        )

        for physical_time, time_index in zip(
            physical_times,
            time_indices,
        ):

            f.write(
                f"  t={physical_time:.4f} "
                f"-> stored index {time_index}\n"
            )


# ================================================================
# MAIN
# ================================================================

def main():
    """Create all worst-case SWE plots."""

    args = parse_args()

    # ------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------

    results_dir = Path(
        args.results_dir
    ).resolve()

    if args.output_dir is None:

        output_dir = (
            results_dir
            /
            "worst_case_plots"
        )

    else:

        output_dir = Path(
            args.output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Required arrays
    # ------------------------------------------------------------

    target_path = (
        results_dir
        /
        "test_ground_truth.npy"
    )

    cfo_path = (
        results_dir
        /
        "cfo_predictions.npy"
    )

    picfo_path = (
        results_dir
        /
        "picfo_predictions.npy"
    )

    for path in [
        target_path,
        cfo_path,
        picfo_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path}"
            )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    target = np.asarray(
        np.load(
            target_path
        ),
        dtype=np.float32,
    )

    cfo_pred = np.asarray(
        np.load(
            cfo_path
        ),
        dtype=np.float32,
    )

    picfo_pred = np.asarray(
        np.load(
            picfo_path
        ),
        dtype=np.float32,
    )

    if (
        target.shape
        !=
        cfo_pred.shape
        or
        target.shape
        !=
        picfo_pred.shape
    ):

        raise ValueError(
            "Target/CFO/PI-CFO arrays "
            "must have identical shapes."
        )

    if target.ndim != 5:

        raise ValueError(
            "Expected shape "
            "(batch,time,nx,ny,3), "
            f"received {target.shape}."
        )

    if target.shape[-1] != 3:

        raise ValueError(
            "Expected SWE channels "
            "[h,hu,hv]."
        )

    # ------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------

    n_time = (
        target.shape[1]
    )

    nx = (
        target.shape[2]
    )

    times = np.linspace(
        0.0,
        1.0,
        n_time,
        dtype=np.float64,
    )

    x_coordinates = np.linspace(
        args.domain_min,
        args.domain_max,
        nx,
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # Worst trajectory
    # ------------------------------------------------------------

    worst_info = select_worst_trajectory(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        selection_mode=args.worst_model,
    )

    worst_index = (
        worst_info[
            "index"
        ]
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "WORST-CASE SWE PLOT"
    )

    print(
        "============================================================"
    )

    print(
        f"Selection mode: "
        f"{args.worst_model}"
    )

    print(
        f"Worst trajectory index: "
        f"{worst_index}"
    )

    print(
        "CFO relative L2: "
        f"{worst_info['cfo_errors'][worst_index]:.8e}"
    )

    print(
        "PI-CFO relative L2: "
        f"{worst_info['picfo_errors'][worst_index]:.8e}"
    )

    # ------------------------------------------------------------
    # Time indices
    # ------------------------------------------------------------

    time_indices = nearest_time_indices(
        times,
        args.times,
    )

    actual_times = [
        float(
            times[
                index
            ]
        )
        for index in time_indices
    ]

    print(
        "\nDisplayed time points:"
    )

    for time_value, index in zip(
        actual_times,
        time_indices,
    ):

        print(
            f"  t={time_value:.4f} "
            f"(index {index})"
        )

    # ------------------------------------------------------------
    # h
    # ------------------------------------------------------------

    y_index = plot_channel(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        trajectory_index=worst_index,
        channel_index=0,
        physical_times=actual_times,
        time_indices=time_indices,
        x_coordinates=x_coordinates,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------
    # hu
    # ------------------------------------------------------------

    plot_channel(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        trajectory_index=worst_index,
        channel_index=1,
        physical_times=actual_times,
        time_indices=time_indices,
        x_coordinates=x_coordinates,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------
    # hv
    # ------------------------------------------------------------

    plot_channel(
        target=target,
        cfo_pred=cfo_pred,
        picfo_pred=picfo_pred,
        trajectory_index=worst_index,
        channel_index=2,
        physical_times=actual_times,
        time_indices=time_indices,
        x_coordinates=x_coordinates,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    save_summary(
        output_dir=output_dir,
        worst_info=worst_info,
        physical_times=actual_times,
        time_indices=time_indices,
        y_index=y_index,
    )

    # ------------------------------------------------------------
    # Done
    # ------------------------------------------------------------

    print(
        "\n"
        "============================================================"
    )

    print(
        "PLOTS CREATED"
    )

    print(
        "============================================================"
    )

    print(
        f"Saved to:\n"
        f"{output_dir}"
    )

    print(
        "\nFiles:"
    )

    print(
        "  worst_case_h_centerline.png"
    )

    print(
        "  worst_case_hu_centerline.png"
    )

    print(
        "  worst_case_hv_centerline.png"
    )

    print(
        "  worst_case_summary.txt"
    )


if __name__ == "__main__":
    main()