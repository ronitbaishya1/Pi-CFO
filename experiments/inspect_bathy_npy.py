"""Inspect saved Bathy-CFO .npy prediction files."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ================================================================
# SETTINGS
# ================================================================

RESULTS_DIR = Path(
    "results/bathy_cfo_full/seed0"
)

TRAJECTORY = 0

TIME_INDEX = 25


# ================================================================
# LOAD FILES
# ================================================================

q_true = np.load(
    RESULTS_DIR
    /
    "test_true.npy"
)

q_pred = np.load(
    RESULTS_DIR
    /
    "test_pred.npy"
)

bathymetry = np.load(
    RESULTS_DIR
    /
    "test_bathymetry.npy"
)


# ================================================================
# PRINT SHAPES
# ================================================================

print(
    "q_true shape:",
    q_true.shape,
)

print(
    "q_pred shape:",
    q_pred.shape,
)

print(
    "bathymetry shape:",
    bathymetry.shape,
)


# ================================================================
# BASIC CHECKS
# ================================================================

print(
    "\nNaN in truth:",
    np.isnan(
        q_true
    ).any(),
)

print(
    "NaN in prediction:",
    np.isnan(
        q_pred
    ).any(),
)

print(
    "Inf in prediction:",
    np.isinf(
        q_pred
    ).any(),
)


# ================================================================
# COMPONENTS
# ================================================================

h_true = q_true[
    ...,
    0
]

hu_true = q_true[
    ...,
    1
]

hv_true = q_true[
    ...,
    2
]

h_pred = q_pred[
    ...,
    0
]

hu_pred = q_pred[
    ...,
    1
]

hv_pred = q_pred[
    ...,
    2
]


# ================================================================
# GLOBAL ERRORS
# ================================================================

absolute_error = np.abs(
    q_pred
    -
    q_true
)

rmse = np.sqrt(
    np.mean(
        (
            q_pred
            -
            q_true
        )
        **
        2
    )
)

relative_l2 = (
    np.linalg.norm(
        (
            q_pred
            -
            q_true
        ).reshape(
            q_true.shape[
                0
            ],
            -1,
        ),
        axis=1,
    )
    /
    np.maximum(
        np.linalg.norm(
            q_true.reshape(
                q_true.shape[
                    0
                ],
                -1,
            ),
            axis=1,
        ),
        1e-12,
    )
)


print(
    "\nGlobal RMSE:",
    rmse,
)

print(
    "\nRelative L2 by trajectory:"
)

for i, value in enumerate(
    relative_l2
):

    print(
        f"Trajectory {i}: "
        f"{value:.8f}"
    )

print(
    "\nMean trajectory Relative L2:",
    np.mean(
        relative_l2
    ),
)


# ================================================================
# COMPONENT-WISE RMSE
# ================================================================

h_rmse = np.sqrt(
    np.mean(
        (
            h_pred
            -
            h_true
        )
        **
        2
    )
)

hu_rmse = np.sqrt(
    np.mean(
        (
            hu_pred
            -
            hu_true
        )
        **
        2
    )
)

hv_rmse = np.sqrt(
    np.mean(
        (
            hv_pred
            -
            hv_true
        )
        **
        2
    )
)


print(
    "\nComponent RMSE:"
)

print(
    "h :",
    h_rmse,
)

print(
    "hu:",
    hu_rmse,
)

print(
    "hv:",
    hv_rmse,
)


# ================================================================
# SELECT ONE TRAJECTORY AND TIME
# ================================================================

h_ref = q_true[
    TRAJECTORY,
    TIME_INDEX,
    :,
    :,
    0,
]

h_model = q_pred[
    TRAJECTORY,
    TIME_INDEX,
    :,
    :,
    0,
]

h_error = np.abs(
    h_model
    -
    h_ref
)

b = bathymetry[
    TRAJECTORY,
    :,
    :,
    0,
]


# ================================================================
# PLOT BATHYMETRY
# ================================================================

plt.figure(
    figsize=(
        6,
        5,
    )
)

plt.imshow(
    b.T,
    origin="lower",
)

plt.colorbar(
    label="b(x,y)"
)

plt.xlabel(
    "x grid index"
)

plt.ylabel(
    "y grid index"
)

plt.title(
    f"Bathymetry — trajectory "
    f"{TRAJECTORY}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    /
    "example_bathymetry.png",
    dpi=200,
)

plt.close()


# ================================================================
# PLOT TRUE h
# ================================================================

plt.figure(
    figsize=(
        6,
        5,
    )
)

plt.imshow(
    h_ref.T,
    origin="lower",
)

plt.colorbar(
    label="h"
)

plt.xlabel(
    "x grid index"
)

plt.ylabel(
    "y grid index"
)

plt.title(
    f"Ground Truth h "
    f"— trajectory {TRAJECTORY}, "
    f"time index {TIME_INDEX}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    /
    "example_h_true.png",
    dpi=200,
)

plt.close()


# ================================================================
# PLOT PREDICTED h
# ================================================================

plt.figure(
    figsize=(
        6,
        5,
    )
)

plt.imshow(
    h_model.T,
    origin="lower",
)

plt.colorbar(
    label="h"
)

plt.xlabel(
    "x grid index"
)

plt.ylabel(
    "y grid index"
)

plt.title(
    f"Bathy-CFO Prediction h "
    f"— trajectory {TRAJECTORY}, "
    f"time index {TIME_INDEX}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    /
    "example_h_prediction.png",
    dpi=200,
)

plt.close()


# ================================================================
# PLOT ABSOLUTE ERROR
# ================================================================

plt.figure(
    figsize=(
        6,
        5,
    )
)

plt.imshow(
    h_error.T,
    origin="lower",
)

plt.colorbar(
    label="|h_pred - h_true|"
)

plt.xlabel(
    "x grid index"
)

plt.ylabel(
    "y grid index"
)

plt.title(
    f"Absolute h Error "
    f"— trajectory {TRAJECTORY}, "
    f"time index {TIME_INDEX}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    /
    "example_h_error.png",
    dpi=200,
)

plt.close()


# ================================================================
# ERROR THROUGH TIME
# ================================================================

trajectory_error_vs_time = np.sqrt(
    np.mean(
        (
            q_pred[
                TRAJECTORY
            ]
            -
            q_true[
                TRAJECTORY
            ]
        )
        **
        2,
        axis=(
            1,
            2,
            3,
        ),
    )
)

time = np.linspace(
    0.0,
    1.0,
    q_true.shape[
        1
    ],
)


plt.figure(
    figsize=(
        7,
        4,
    )
)

plt.plot(
    time,
    trajectory_error_vs_time,
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    f"Prediction Error Through Time "
    f"— trajectory {TRAJECTORY}"
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR
    /
    "example_error_vs_time.png",
    dpi=200,
)

plt.close()


# ================================================================
# FINISH
# ================================================================

print(
    "\nPlots saved in:"
)

print(
    RESULTS_DIR.resolve()
)