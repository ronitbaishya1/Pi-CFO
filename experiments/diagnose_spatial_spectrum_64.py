"""
64x64 spatial-spectrum diagnostic for:

    1. PyClaw reference
    2. Bathymetry-conditioned CFO
    3. Final Geometry-U-FNO + WB-Bed-PI-CFO

Purpose
-------
Determine whether the jagged 64x64 hu/hv profiles are caused by
excessive high-spatial-frequency content in:

    A. the predicted state,
    B. the isolated terrain response,
    C. the learned vector field itself.

Outputs
-------
01_total_field_radial_spectrum.png
02_terrain_effect_radial_spectrum.png
03_vector_field_radial_spectrum.png
04_high_frequency_fraction.png
05_fft_maps_momentum.png

spectral_metrics.csv
spectral_metrics.md

No training is performed.
No checkpoint is changed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =====================================================================
# PROJECT IMPORTS
# =====================================================================

from cfo import ContinuousFlowOperator

from models.factory import build_model

from models.geometry_ufno import (
    GeometryUFNO2d,
)

from train import (
    init_cfo_train_state,
)

from utils.checkpoints import (
    load_train_state,
)

from wb_bathy_bed_pi_cfo import (
    WellBalancedBathymetryBedPICFO,
)

import experiments.plot_wb_bed_diagnostics as diag


# =====================================================================
# CONSTANTS
# =====================================================================

RESOLUTION = 64

X_MIN = -2.5
X_MAX = 2.5

Y_MIN = -2.5
Y_MAX = 2.5

DX = (
    X_MAX
    -
    X_MIN
) / RESOLUTION

DY = (
    Y_MAX
    -
    Y_MIN
) / RESOLUTION

GRAVITY = 1.0


CHANNEL_NAMES = [
    "h",
    "hu",
    "hv",
]


MODEL_KEYS = [
    "pyclaw",
    "cfo",
    "final",
]


MODEL_LABELS = {
    "pyclaw":
        "PyClaw",

    "cfo":
        "Bathy-CFO",

    "final":
        "Final Geometry-U-FNO",
}


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Diagnose high-spatial-frequency content "
            "in 64x64 SWE fields."
        )
    )

    # -----------------------------------------------------------------
    # DATA
    # -----------------------------------------------------------------

    parser.add_argument(
        "--counterfactual-data",
        type=str,
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--case",
        type=str,
        default="hill_center",
        choices=[
            "hill_left",
            "hill_center",
            "hill_right",
        ],
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # BATHY-CFO CHECKPOINT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--cfo-ckpt-dir",
        type=str,
        default=(
            "checkpoints/"
            "bathy_cfo_full/"
            "seed0/"
            "best"
        ),
    )

    parser.add_argument(
        "--cfo-prefix",
        type=str,
        default="bathy_cfo",
    )

    # -----------------------------------------------------------------
    # FINAL 64x64 GEOMETRY-U-FNO CHECKPOINT
    #
    # If omitted, the script searches res64 automatically and first
    # looks for:
    #
    #     pde_0p01__bed_0p7__wb_0p1
    #
    # If your checkpoint is elsewhere, simply provide --final-ckpt.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--final-ckpt",
        type=str,
        default=None,
        help=(
            "Optional explicit 64x64 Geometry-U-FNO "
            "checkpoint-manager path."
        ),
    )

    # -----------------------------------------------------------------
    # FINAL LOSS CONFIGURATION
    # -----------------------------------------------------------------

    parser.add_argument(
        "--lambda-pde",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--lambda-bed",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--lambda-wb",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--wb-eta0",
        type=float,
        default=1.5,
    )

    # -----------------------------------------------------------------
    # GEOMETRY-U-FNO ARCHITECTURE
    # -----------------------------------------------------------------

    parser.add_argument(
        "--modes1",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--modes2",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-u-blocks",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--geometry-width",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--geometry-depth",
        type=int,
        default=2,
    )

    # -----------------------------------------------------------------
    # OUTPUT
    # -----------------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "results/"
            "spatial_spectrum_64"
        ),
    )

    return parser.parse_args()


# =====================================================================
# PATH HELPERS
# =====================================================================

def resolve_path(
    value,
):

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            /
            path
        )

    return path.resolve()


# =====================================================================
# CONFIGURE EXISTING DIAGNOSTICS
# =====================================================================

def configure_diag():

    diag.NX = RESOLUTION
    diag.NY = RESOLUTION

    diag.DX = DX
    diag.DY = DY

    diag.X_MIN = X_MIN
    diag.X_MAX = X_MAX

    diag.Y_MIN = Y_MIN
    diag.Y_MAX = Y_MAX


# =====================================================================
# BUILD BATHY-CFO
# =====================================================================

def build_cfo_method():

    input_shape = (
        RESOLUTION,
        RESOLUTION,
        3,
    )

    model = build_model(
        "FNO2d",
        input_shape,
        use_condition=True,
    )

    return ContinuousFlowOperator(
        model=model,

        input_shape=input_shape,

        gamma=1.0e-5,

        spline_type="quintic",

        use_condition=True,

        condition_shape=(
            RESOLUTION,
            RESOLUTION,
            1,
        ),
    )


# =====================================================================
# BUILD FINAL GEOMETRY-U-FNO
# =====================================================================

def build_final_method(
    args,
):

    model = GeometryUFNO2d(
        num_channels=3,

        modes1=args.modes1,
        modes2=args.modes2,

        width=args.width,

        num_blocks=args.num_blocks,
        num_u_blocks=args.num_u_blocks,

        geometry_width=args.geometry_width,
        geometry_depth=args.geometry_depth,

        dx=DX,
        dy=DY,

        include_gradient_magnitude=False,

        use_time=True,
    )

    return WellBalancedBathymetryBedPICFO(
        model=model,

        input_shape=(
            RESOLUTION,
            RESOLUTION,
            3,
        ),

        condition_shape=(
            RESOLUTION,
            RESOLUTION,
            1,
        ),

        gamma=1.0e-5,

        spline_type="quintic",

        lambda_pde=args.lambda_pde,

        lambda_bed=args.lambda_bed,

        lambda_wb=args.lambda_wb,

        wb_eta0=args.wb_eta0,

        dx=DX,
        dy=DY,

        gravity=GRAVITY,
    )


# =====================================================================
# RESTORE BATHY-CFO
# =====================================================================

def restore_cfo_state(
    method,
    args,
):

    checkpoint_dir = resolve_path(
        args.cfo_ckpt_dir
    )

    if not checkpoint_dir.exists():

        raise FileNotFoundError(
            "Bathy-CFO checkpoint directory "
            "was not found:\n"
            f"{checkpoint_dir}"
        )

    target_state = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target_state,

        ckpt_dir=str(
            checkpoint_dir
        ),

        prefix=args.cfo_prefix,

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# FIND FINAL CHECKPOINT
# =====================================================================

def locate_final_checkpoint(
    args,
):

    # -------------------------------------------------------------
    # Explicit checkpoint always takes priority.
    # -------------------------------------------------------------

    if args.final_ckpt is not None:

        checkpoint_path = resolve_path(
            args.final_ckpt
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                "Explicit final checkpoint "
                "was not found:\n"
                f"{checkpoint_path}"
            )

        return checkpoint_path

    # -------------------------------------------------------------
    # Otherwise search the existing 64x64 checkpoint tree.
    # -------------------------------------------------------------

    search_root = (
        PROJECT_ROOT
        /
        "checkpoints"
        /
        "geometry_wb_bed_pi"
        /
        "geometry_ufno"
        /
        "res64"
    )

    if not search_root.exists():

        raise FileNotFoundError(
            "64x64 Geometry-U-FNO checkpoint "
            "root was not found:\n"
            f"{search_root}"
        )

    candidates = sorted(
        path
        for path
        in search_root.rglob(
            "best"
        )
        if path.is_dir()
    )

    # -------------------------------------------------------------
    # Prefer our selected 64x64 configuration.
    # -------------------------------------------------------------

    preferred_token = (
        "pde_0p01__bed_0p7__wb_0p1"
    )

    preferred = [
        path
        for path
        in candidates
        if preferred_token
        in str(
            path
        )
    ]

    if preferred:
        return preferred[
            0
        ]

    # -------------------------------------------------------------
    # Old/default path fallback.
    # -------------------------------------------------------------

    standard = (
        search_root
        /
        "lamwb_0p1"
        /
        "seed0"
        /
        "best"
    )

    if standard.exists():

        print()
        print(
            "WARNING:"
        )

        print(
            "Preferred 0.01/0.7/0.1 checkpoint "
            "was not found."
        )

        print(
            "Using standard res64 checkpoint:"
        )

        print(
            standard
        )

        return standard

    # -------------------------------------------------------------
    # If there is exactly one, it is still unambiguous.
    # -------------------------------------------------------------

    if len(
        candidates
    ) == 1:

        print()
        print(
            "WARNING:"
        )

        print(
            "Using the only res64 /best "
            "checkpoint found:"
        )

        print(
            candidates[
                0
            ]
        )

        return candidates[
            0
        ]

    raise FileNotFoundError(
        "Could not uniquely identify the "
        "final 64x64 checkpoint.\n"
        "Pass it explicitly with --final-ckpt.\n"
        f"Candidates: {[str(x) for x in candidates]}"
    )


# =====================================================================
# RESTORE FINAL MODEL
# =====================================================================

def restore_final_state(
    method,
    args,
):

    checkpoint_path = (
        locate_final_checkpoint(
            args
        )
    )

    print()
    print(
        "Using final Geometry-U-FNO checkpoint:"
    )

    print(
        checkpoint_path
    )

    target_state = init_cfo_train_state(
        method,

        seed=0,

        learning_rate=1.0e-4,

        beta1=0.9,
        beta2=0.99,
    )

    return load_train_state(
        target_state,

        ckpt_dir=str(
            checkpoint_path.parent
        ),

        prefix=(
            checkpoint_path.name
        ),

        step=None,

        max_to_keep=1,
    )


# =====================================================================
# SPECTRAL WINDOW
# =====================================================================

def hann_window(
    shape,
):

    nx, ny = shape

    window_x = np.hanning(
        nx
    )

    window_y = np.hanning(
        ny
    )

    return (
        window_x[
            :,
            None
        ]
        *
        window_y[
            None,
            :
        ]
    )


# =====================================================================
# NORMALIZED FREQUENCY GRID
# =====================================================================

def normalized_frequency_radius(
    shape,
    dx,
    dy,
):

    nx, ny = shape

    fx = (
        np.fft.fftshift(
            np.fft.fftfreq(
                nx,
                d=dx,
            )
        )
        /
        (
            1.0
            /
            (
                2.0
                *
                dx
            )
        )
    )

    fy = (
        np.fft.fftshift(
            np.fft.fftfreq(
                ny,
                d=dy,
            )
        )
        /
        (
            1.0
            /
            (
                2.0
                *
                dy
            )
        )
    )

    FX, FY = np.meshgrid(
        fx,
        fy,
        indexing="ij",
    )

    return np.sqrt(
        FX**2
        +
        FY**2
    )


# =====================================================================
# 2D POWER SPECTRUM
# =====================================================================

def power_spectrum_2d(
    field,
):

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    if field.ndim != 2:

        raise ValueError(
            "Expected 2-D field, "
            f"got {field.shape}."
        )

    # -------------------------------------------------------------
    # Remove zero-frequency mean component.
    # -------------------------------------------------------------

    centered = (
        field
        -
        np.mean(
            field
        )
    )

    # -------------------------------------------------------------
    # Hanning window reduces edge spectral leakage.
    # -------------------------------------------------------------

    windowed = (
        centered
        *
        hann_window(
            field.shape
        )
    )

    fourier = (
        np.fft.fftshift(
            np.fft.fft2(
                windowed
            )
        )
    )

    return (
        np.abs(
            fourier
        )
        ** 2
    )


# =====================================================================
# RADIAL SPECTRUM
# =====================================================================

def radial_spectrum(
    field,
    dx,
    dy,
    num_bins=32,
):

    power = power_spectrum_2d(
        field
    )

    rho = normalized_frequency_radius(
        field.shape,
        dx,
        dy,
    )

    rho_flat = rho.ravel()

    power_flat = (
        power.ravel()
    )

    bins = np.linspace(
        0.0,
        float(
            rho_flat.max()
        ),
        num_bins
        +
        1,
    )

    centers = (
        0.5
        *
        (
            bins[
                :-1
            ]
            +
            bins[
                1:
            ]
        )
    )

    values = np.full(
        num_bins,
        np.nan,
        dtype=np.float64,
    )

    for index in range(
        num_bins
    ):

        if (
            index
            ==
            num_bins
            -
            1
        ):

            mask = (
                (
                    rho_flat
                    >=
                    bins[
                        index
                    ]
                )
                &
                (
                    rho_flat
                    <=
                    bins[
                        index
                        +
                        1
                    ]
                )
            )

        else:

            mask = (
                (
                    rho_flat
                    >=
                    bins[
                        index
                    ]
                )
                &
                (
                    rho_flat
                    <
                    bins[
                        index
                        +
                        1
                    ]
                )
            )

        if np.any(
            mask
        ):

            values[
                index
            ] = np.mean(
                power_flat[
                    mask
                ]
            )

    total = np.nansum(
        values
    )

    if total > 0.0:

        values = (
            values
            /
            total
        )

    return (
        centers,
        values,
    )


# =====================================================================
# SPECTRAL METRICS
# =====================================================================

def spectrum_metrics(
    field,
    dx,
    dy,
):

    field = np.asarray(
        field,
        dtype=np.float64,
    )

    power = power_spectrum_2d(
        field
    )

    rho = normalized_frequency_radius(
        field.shape,
        dx,
        dy,
    )

    total_power = float(
        np.sum(
            power
        )
    )

    if total_power <= 1.0e-30:

        return {
            "low_fraction":
                0.0,

            "mid_fraction":
                0.0,

            "high_fraction":
                0.0,

            "spectral_centroid":
                0.0,

            "roughness":
                0.0,
        }

    # -------------------------------------------------------------
    # Normalized frequency regions.
    #
    # low  : rho < 0.33
    # mid  : 0.33 <= rho < 0.66
    # high : rho >= 0.66
    # -------------------------------------------------------------

    low_mask = (
        rho
        <
        0.33
    )

    mid_mask = (
        (
            rho
            >=
            0.33
        )
        &
        (
            rho
            <
            0.66
        )
    )

    high_mask = (
        rho
        >=
        0.66
    )

    low_fraction = float(
        np.sum(
            power[
                low_mask
            ]
        )
        /
        total_power
    )

    mid_fraction = float(
        np.sum(
            power[
                mid_mask
            ]
        )
        /
        total_power
    )

    high_fraction = float(
        np.sum(
            power[
                high_mask
            ]
        )
        /
        total_power
    )

    spectral_centroid = float(
        np.sum(
            rho
            *
            power
        )
        /
        total_power
    )

    # -------------------------------------------------------------
    # Physical-space roughness metric.
    # -------------------------------------------------------------

    gradient_x = np.gradient(
        field,
        dx,
        axis=0,
        edge_order=2,
    )

    gradient_y = np.gradient(
        field,
        dy,
        axis=1,
        edge_order=2,
    )

    gradient_energy = float(
        np.mean(
            gradient_x**2
            +
            gradient_y**2
        )
    )

    field_energy = float(
        np.mean(
            field**2
        )
    )

    roughness = (
        gradient_energy
        /
        max(
            field_energy,
            1.0e-30,
        )
    )

    return {
        "low_fraction":
            low_fraction,

        "mid_fraction":
            mid_fraction,

        "high_fraction":
            high_fraction,

        "spectral_centroid":
            spectral_centroid,

        "roughness":
            roughness,
    }


# =====================================================================
# PYCLAW TEMPORAL DERIVATIVE
# =====================================================================

def pyclaw_time_derivative(
    q,
    time,
    index,
):

    q = np.asarray(
        q,
        dtype=np.float64,
    )

    time = np.asarray(
        time,
        dtype=np.float64,
    )

    if (
        index <= 0
        or
        index
        >=
        len(
            time
        )
        -
        1
    ):

        raise ValueError(
            "Choose an interior plot time, "
            "for example t=0.50."
        )

    delta_time = float(
        time[
            index
            +
            1
        ]
        -
        time[
            index
            -
            1
        ]
    )

    return (
        q[
            index
            +
            1
        ]
        -
        q[
            index
            -
            1
        ]
    ) / delta_time


# =====================================================================
# RADIAL SPECTRUM PLOT
# =====================================================================

def plot_radial_spectra(
    fields,
    title,
    output_path,
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            18,
            5.5,
        ),
    )

    for channel_index, ax in enumerate(
        axes
    ):

        for model_key in MODEL_KEYS:

            rho, spectrum = radial_spectrum(
                fields[
                    model_key
                ][
                    ...,
                    channel_index
                ],

                DX,
                DY,
            )

            ax.semilogy(
                rho,

                np.maximum(
                    spectrum,
                    1.0e-14,
                ),

                linewidth=2,

                label=(
                    MODEL_LABELS[
                        model_key
                    ]
                ),
            )

        ax.axvline(
            0.66,

            linestyle="--",

            linewidth=1.2,

            label=(
                "high-frequency threshold"
                if
                channel_index == 0
                else
                None
            ),
        )

        ax.set_title(
            CHANNEL_NAMES[
                channel_index
            ]
        )

        ax.set_xlabel(
            "Normalized radial spatial frequency"
        )

        ax.set_ylabel(
            "Normalized radial spectral energy"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

    fig.suptitle(
        title,
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# BUILD METRIC TABLE
# =====================================================================

def build_metric_rows(
    groups,
):

    rows = []

    for kind, fields in groups.items():

        for model_key in MODEL_KEYS:

            for (
                channel_index,
                channel_name,
            ) in enumerate(
                CHANNEL_NAMES
            ):

                metrics = spectrum_metrics(
                    fields[
                        model_key
                    ][
                        ...,
                        channel_index
                    ],

                    DX,
                    DY,
                )

                rows.append(
                    {
                        "kind":
                            kind,

                        "model":
                            model_key,

                        "model_label":
                            MODEL_LABELS[
                                model_key
                            ],

                        "channel":
                            channel_name,

                        **metrics,
                    }
                )

    return rows


# =====================================================================
# HIGH-FREQUENCY BAR PLOT
# =====================================================================

def plot_high_frequency_bars(
    rows,
    output_path,
):

    kinds = [
        "total_field",
        "terrain_effect",
        "vector_field",
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            12,
            12,
        ),
    )

    x = np.arange(
        3
    )

    width = 0.24

    for ax, kind in zip(
        axes,
        kinds,
    ):

        for model_index, model_key in enumerate(
            MODEL_KEYS
        ):

            values = []

            for channel_name in CHANNEL_NAMES:

                row = next(
                    row
                    for row
                    in rows
                    if (
                        row[
                            "kind"
                        ]
                        ==
                        kind
                        and
                        row[
                            "model"
                        ]
                        ==
                        model_key
                        and
                        row[
                            "channel"
                        ]
                        ==
                        channel_name
                    )
                )

                values.append(
                    row[
                        "high_fraction"
                    ]
                )

            ax.bar(
                x
                +
                (
                    model_index
                    -
                    1
                )
                *
                width,

                values,

                width=width,

                label=(
                    MODEL_LABELS[
                        model_key
                    ]
                ),
            )

        ax.set_xticks(
            x,
            CHANNEL_NAMES,
        )

        ax.set_ylabel(
            "High-frequency energy fraction"
        )

        ax.set_title(
            kind
            .replace(
                "_",
                " ",
            )
            .title()
        )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        ax.legend()

    fig.suptitle(
        "64x64 high-spatial-frequency energy",
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# MOMENTUM FFT MAPS
# =====================================================================

def plot_fft_maps_momentum(
    fields,
    output_path,
):

    frequency_x = (
        np.fft.fftshift(
            np.fft.fftfreq(
                RESOLUTION,
                d=DX,
            )
        )
        /
        (
            1.0
            /
            (
                2.0
                *
                DX
            )
        )
    )

    frequency_y = (
        np.fft.fftshift(
            np.fft.fftfreq(
                RESOLUTION,
                d=DY,
            )
        )
        /
        (
            1.0
            /
            (
                2.0
                *
                DY
            )
        )
    )

    extent = [
        float(
            frequency_y[
                0
            ]
        ),

        float(
            frequency_y[
                -1
            ]
        ),

        float(
            frequency_x[
                0
            ]
        ),

        float(
            frequency_x[
                -1
            ]
        ),
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )

    # -------------------------------------------------------------
    # Only momentum channels:
    #
    # row 0 = hu
    # row 1 = hv
    # -------------------------------------------------------------

    for row, channel_index in enumerate(
        [
            1,
            2,
        ]
    ):

        maps = [
            np.log10(
                power_spectrum_2d(
                    fields[
                        model_key
                    ][
                        ...,
                        channel_index
                    ]
                )
                +
                1.0e-16
            )
            for model_key
            in MODEL_KEYS
        ]

        vmin = min(
            float(
                value.min()
            )
            for value
            in maps
        )

        vmax = max(
            float(
                value.max()
            )
            for value
            in maps
        )

        for col, model_key in enumerate(
            MODEL_KEYS
        ):

            image = axes[
                row,
                col
            ].imshow(
                maps[
                    col
                ],

                origin="lower",

                extent=extent,

                vmin=vmin,
                vmax=vmax,

                aspect="equal",
            )

            axes[
                row,
                col
            ].set_title(
                (
                    MODEL_LABELS[
                        model_key
                    ]
                    +
                    "\n"
                    +
                    CHANNEL_NAMES[
                        channel_index
                    ]
                )
            )

            axes[
                row,
                col
            ].set_xlabel(
                "normalized fy"
            )

            axes[
                row,
                col
            ].set_ylabel(
                "normalized fx"
            )

            fig.colorbar(
                image,

                ax=axes[
                    row,
                    col
                ],

                shrink=0.8,

                label=(
                    "log10 spectral power"
                ),
            )

    fig.suptitle(
        "64x64 momentum FFT maps at selected time",
        fontsize=17,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =====================================================================
# SAVE TABLES
# =====================================================================

def save_tables(
    rows,
    output_dir,
):

    fieldnames = [
        "kind",
        "model",
        "model_label",
        "channel",
        "low_fraction",
        "mid_fraction",
        "high_fraction",
        "spectral_centroid",
        "roughness",
    ]

    # -----------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------

    with open(
        output_dir
        /
        "spectral_metrics.csv",

        "w",

        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # -----------------------------------------------------------------
    # MARKDOWN
    # -----------------------------------------------------------------

    with open(
        output_dir
        /
        "spectral_metrics.md",

        "w",
    ) as file:

        file.write(
            "# 64x64 spatial-spectrum diagnostic\n\n"
        )

        file.write(
            "High-frequency band: "
            "normalized radial frequency >= 0.66.\n\n"
        )

        file.write(
            "| Kind | Model | Channel | "
            "Low | Mid | High | "
            "Centroid | Roughness |\n"
        )

        file.write(
            "| --- | --- | --- | ---: | ---: | "
            "---: | ---: | ---: |\n"
        )

        for row in rows:

            file.write(
                f"| {row['kind']} "
                f"| {row['model_label']} "
                f"| {row['channel']} "
                f"| {row['low_fraction']:.6f} "
                f"| {row['mid_fraction']:.6f} "
                f"| {row['high_fraction']:.6f} "
                f"| {row['spectral_centroid']:.6f} "
                f"| {row['roughness']:.6e} |\n"
            )


# =====================================================================
# CONSOLE SUMMARY
# =====================================================================

def print_summary(
    rows,
):

    print()
    print(
        "=" * 78
    )

    print(
        "HIGH-FREQUENCY SUMMARY"
    )

    print(
        "=" * 78
    )

    for kind in [
        "total_field",
        "terrain_effect",
        "vector_field",
    ]:

        print()
        print(
            kind.upper()
        )

        print(
            "-" * 78
        )

        for channel_name in CHANNEL_NAMES:

            by_model = {
                row[
                    "model"
                ]:
                    row

                for row
                in rows

                if (
                    row[
                        "kind"
                    ]
                    ==
                    kind

                    and

                    row[
                        "channel"
                    ]
                    ==
                    channel_name
                )
            }

            reference = (
                by_model[
                    "pyclaw"
                ][
                    "high_fraction"
                ]
            )

            print(
                f"{channel_name}:"
            )

            for model_key in MODEL_KEYS:

                value = (
                    by_model[
                        model_key
                    ][
                        "high_fraction"
                    ]
                )

                ratio = (
                    value
                    /
                    reference

                    if
                    reference
                    >
                    1.0e-15

                    else
                    np.nan
                )

                print(
                    f"  "
                    f"{MODEL_LABELS[model_key]:24s} "
                    f"high={value:.6f}  "
                    f"vs-PyClaw={ratio:.3f}x"
                )

    print(
        "=" * 78
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    configure_diag()

    # -----------------------------------------------------------------
    # BASIC CONFIGURATION
    # -----------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        "64x64 SPATIAL-SPECTRUM DIAGNOSTIC"
    )

    print(
        "=" * 78
    )

    print(
        f"Resolution : "
        f"{RESOLUTION} x {RESOLUTION}"
    )

    print(
        f"dx, dy     : "
        f"{DX:.10f}, {DY:.10f}"
    )

    print(
        f"Case       : "
        f"{args.case}"
    )

    print(
        f"Plot time  : "
        f"{args.plot_time:.4f}"
    )

    print(
        f"RK4 steps  : "
        f"{args.steps_per_segment}"
    )

    print(
        "Losses     : "
        f"PDE={args.lambda_pde}, "
        f"bed={args.lambda_bed}, "
        f"WB={args.lambda_wb}"
    )

    print(
        "=" * 78
    )

    # -----------------------------------------------------------------
    # BUILD MODELS
    # -----------------------------------------------------------------

    cfo_method = (
        build_cfo_method()
    )

    final_method = (
        build_final_method(
            args
        )
    )

    # -----------------------------------------------------------------
    # RESTORE CHECKPOINTS
    # -----------------------------------------------------------------

    print()
    print(
        "Restoring Bathy-CFO..."
    )

    cfo_state = (
        restore_cfo_state(
            cfo_method,
            args,
        )
    )

    print(
        "Restoring final 64x64 "
        "Geometry-U-FNO..."
    )

    final_state = (
        restore_final_state(
            final_method,
            args,
        )
    )

    # -----------------------------------------------------------------
    # OUTPUT DIRECTORY
    # -----------------------------------------------------------------

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # LOAD COUNTERFACTUAL DATA
    # -----------------------------------------------------------------

    archive = diag.H5Archive(
        args.counterfactual_data
    )

    try:

        data = archive.get_pair(
            args.case,

            sample_index=(
                args.sample_index
            ),
        )

        time = np.asarray(
            data[
                "time"
            ],
            dtype=np.float64,
        )

        index = int(
            np.argmin(
                np.abs(
                    time
                    -
                    args.plot_time
                )
            )
        )

        if (
            index <= 0
            or
            index
            >=
            len(
                time
            )
            -
            1
        ):

            raise ValueError(
                "Selected plot-time is too close "
                "to a trajectory boundary."
            )

        time_value = float(
            time[
                index
            ]
        )

        print()
        print(
            "Using time index",
            index,
            "at t =",
            f"{time_value:.6f}",
        )

        # =============================================================
        # MODEL ROLLOUTS
        # =============================================================

        print()
        print(
            "Generating Bathy-CFO rollout..."
        )

        (
            cfo_hill,
            cfo_flat,
        ) = diag.model_pair_rollout(
            cfo_method,
            cfo_state,
            data,

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        print(
            "Generating final "
            "Geometry-U-FNO rollout..."
        )

        (
            final_hill,
            final_flat,
        ) = diag.model_pair_rollout(
            final_method,
            final_state,
            data,

            steps_per_segment=(
                args.steps_per_segment
            ),
        )

        # =============================================================
        # GROUP 1 — TOTAL FIELDS
        # =============================================================

        total_fields = {
            "pyclaw":
                np.asarray(
                    data[
                        "terrain_q"
                    ][
                        index
                    ],

                    dtype=np.float64,
                ),

            "cfo":
                np.asarray(
                    cfo_hill[
                        index
                    ],

                    dtype=np.float64,
                ),

            "final":
                np.asarray(
                    final_hill[
                        index
                    ],

                    dtype=np.float64,
                ),
        }

        # =============================================================
        # GROUP 2 — TERRAIN-INDUCED EFFECT
        #
        # q_terrain - q_flat
        # =============================================================

        terrain_effect = {
            "pyclaw":
                np.asarray(
                    data[
                        "terrain_q"
                    ][
                        index
                    ]
                    -
                    data[
                        "flat_q"
                    ][
                        index
                    ],

                    dtype=np.float64,
                ),

            "cfo":
                np.asarray(
                    cfo_hill[
                        index
                    ]
                    -
                    cfo_flat[
                        index
                    ],

                    dtype=np.float64,
                ),

            "final":
                np.asarray(
                    final_hill[
                        index
                    ]
                    -
                    final_flat[
                        index
                    ],

                    dtype=np.float64,
                ),
        }

        # =============================================================
        # GROUP 3 — VECTOR FIELD ON IDENTICAL PYCLAW STATE
        #
        # This is the most important diagnostic.
        #
        # All neural models receive exactly the same:
        #
        # q = PyClaw q(t=0.5)
        # b = Gaussian bathymetry
        # t = 0.5
        #
        # Therefore differences here originate directly from the
        # learned vector field rather than accumulated RK4 error.
        # =============================================================

        reference_state = np.asarray(
            data[
                "terrain_q"
            ][
                index
            ],

            dtype=np.float32,
        )

        bathymetry = np.asarray(
            data[
                "b"
            ],

            dtype=np.float32,
        )

        pyclaw_q_t = (
            pyclaw_time_derivative(
                data[
                    "terrain_q"
                ],

                time,

                index,
            )
        )

        cfo_q_t = (
            diag.model_vector_field(
                cfo_method,
                cfo_state,

                reference_state,
                bathymetry,

                time_value,
            )
        )

        final_q_t = (
            diag.model_vector_field(
                final_method,
                final_state,

                reference_state,
                bathymetry,

                time_value,
            )
        )

        vector_fields = {
            "pyclaw":
                np.asarray(
                    pyclaw_q_t,
                    dtype=np.float64,
                ),

            "cfo":
                np.asarray(
                    cfo_q_t,
                    dtype=np.float64,
                ),

            "final":
                np.asarray(
                    final_q_t,
                    dtype=np.float64,
                ),
        }

        # =============================================================
        # METRIC TABLE
        # =============================================================

        groups = {
            "total_field":
                total_fields,

            "terrain_effect":
                terrain_effect,

            "vector_field":
                vector_fields,
        }

        metric_rows = (
            build_metric_rows(
                groups
            )
        )

        save_tables(
            metric_rows,
            output_dir,
        )

        # =============================================================
        # FIGURE 01 — TOTAL FIELD
        # =============================================================

        plot_radial_spectra(
            total_fields,

            (
                "64x64 total-field radial spectra "
                f"at t={time_value:.2f}"
            ),

            output_dir
            /
            "01_total_field_radial_spectrum.png",
        )

        # =============================================================
        # FIGURE 02 — TERRAIN EFFECT
        # =============================================================

        plot_radial_spectra(
            terrain_effect,

            (
                "64x64 terrain-induced radial spectra "
                f"at t={time_value:.2f}"
            ),

            output_dir
            /
            "02_terrain_effect_radial_spectrum.png",
        )

        # =============================================================
        # FIGURE 03 — VECTOR FIELD
        # =============================================================

        plot_radial_spectra(
            vector_fields,

            (
                "64x64 vector-field radial spectra "
                f"at t={time_value:.2f}"
            ),

            output_dir
            /
            "03_vector_field_radial_spectrum.png",
        )

        # =============================================================
        # FIGURE 04 — HIGH FREQUENCY FRACTIONS
        # =============================================================

        plot_high_frequency_bars(
            metric_rows,

            output_dir
            /
            "04_high_frequency_fraction.png",
        )

        # =============================================================
        # FIGURE 05 — 2D MOMENTUM FFT MAPS
        # =============================================================

        plot_fft_maps_momentum(
            total_fields,

            output_dir
            /
            "05_fft_maps_momentum.png",
        )

        # =============================================================
        # CONSOLE SUMMARY
        # =============================================================

        print_summary(
            metric_rows
        )

        print()
        print(
            "Saved diagnostic outputs to:"
        )

        print(
            output_dir
        )

        print()

    finally:

        archive.close()


if __name__ == "__main__":
    main()