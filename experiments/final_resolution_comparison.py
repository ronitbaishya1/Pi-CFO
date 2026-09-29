from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Patch

from models.geometry_ufno import GeometryUFNO2d
from wb_bathy_bed_pi_cfo import WellBalancedBathymetryBedPICFO
from train import init_cfo_train_state
from utils.checkpoints import load_train_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]

XMIN = -2.5
XMAX = 2.5
YMIN = -2.5
YMAX = 2.5

GRAVITY = 1.0


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gaussian64",
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_64.h5"
        ),
    )

    parser.add_argument(
        "--gaussian128",
        default=(
            "data/shallow_water_bathy/"
            "gaussian_counterfactual_128.h5"
        ),
    )

    parser.add_argument(
        "--ood64",
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_64.h5"
        ),
    )

    parser.add_argument(
        "--ood128",
        default=(
            "data/shallow_water_bathy/"
            "swe_bathy_ood_128.h5"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/rollout_stable/"
            "geometry_ufno/res64/"
            "K2_lamroll_0p1_lamhf_0/"
            "seed0/best"
        ),
    )

    parser.add_argument(
        "--plot-time",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/final_resolution_comparison/"
            "64_vs_zero128_vs_128"
        ),
    )

    return parser.parse_args()


def resolve(path):

    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# =====================================================================
# DATA
# =====================================================================

def load_case(path, case_name):

    path = resolve(path)

    with h5py.File(path, "r") as h5:

        q = np.asarray(
            h5[f"cases/{case_name}/q"],
            dtype=np.float32,
        )

        b = np.asarray(
            h5[f"cases/{case_name}/bathymetry"],
            dtype=np.float32,
        )

        time = np.asarray(
            h5["time"],
            dtype=np.float32,
        )

        x = np.asarray(
            h5["x"],
            dtype=np.float32,
        )

        y = np.asarray(
            h5["y"],
            dtype=np.float32,
        )

    if b.ndim == 3 and b.shape[-1] == 1:
        b = b[..., 0]

    return {
        "q": q,
        "b": b,
        "time": time,
        "x": x,
        "y": y,
    }


# =====================================================================
# INTERPOLATE 64 -> 128
# =====================================================================

def interp2(field, x0, y0, x1, y1):

    temp = np.empty(
        (
            len(x1),
            len(y0),
        ),
        dtype=np.float64,
    )

    for j in range(len(y0)):

        temp[:, j] = np.interp(
            x1,
            x0,
            field[:, j],
        )

    result = np.empty(
        (
            len(x1),
            len(y1),
        ),
        dtype=np.float64,
    )

    for i in range(len(x1)):

        result[i, :] = np.interp(
            y1,
            y0,
            temp[i, :],
        )

    return result


def interpolate_state(q, x0, y0, x1, y1):

    nt = q.shape[0]
    nc = q.shape[-1]

    output = np.empty(
        (
            nt,
            len(x1),
            len(y1),
            nc,
        ),
        dtype=np.float32,
    )

    for t in range(nt):

        for c in range(nc):

            output[t, ..., c] = interp2(
                q[t, ..., c],
                x0,
                y0,
                x1,
                y1,
            )

    return output


def interpolate_bed(b, x0, y0, x1, y1):

    return interp2(
        b,
        x0,
        y0,
        x1,
        y1,
    ).astype(
        np.float32
    )


# =====================================================================
# MODEL
# =====================================================================

def build_zero_shot_model(checkpoint):

    resolution = 128

    dx = (
        XMAX
        -
        XMIN
    ) / resolution

    dy = (
        YMAX
        -
        YMIN
    ) / resolution

    model = GeometryUFNO2d(
        num_channels=3,
        modes1=12,
        modes2=12,
        width=64,
        num_blocks=4,
        num_u_blocks=2,
        geometry_width=16,
        geometry_depth=2,
        dx=dx,
        dy=dy,
        include_gradient_magnitude=False,
        use_time=True,
    )

    method = WellBalancedBathymetryBedPICFO(
        model=model,
        input_shape=(
            resolution,
            resolution,
            3,
        ),
        condition_shape=(
            resolution,
            resolution,
            1,
        ),
        gamma=1.0e-5,
        spline_type="quintic",
        lambda_pde=0.01,
        lambda_bed=0.70,
        lambda_wb=0.10,
        wb_eta0=1.5,
        dx=dx,
        dy=dy,
        gravity=1.0,
    )

    state = init_cfo_train_state(
        method,
        seed=0,
        learning_rate=1.0e-4,
        beta1=0.9,
        beta2=0.99,
    )

    checkpoint = resolve(checkpoint)

    state = load_train_state(
        state,
        ckpt_dir=str(
            checkpoint.parent
        ),
        prefix=(
            checkpoint.name
        ),
        step=None,
        max_to_keep=1,
    )

    return method, state


# =====================================================================
# ROLLOUT
# =====================================================================

def normalize_prediction(prediction, nt):

    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if prediction.ndim == 5:

        if (
            prediction.shape[0] == nt
            and
            prediction.shape[1] == 1
        ):

            return prediction[:, 0]

        if (
            prediction.shape[0] == 1
            and
            prediction.shape[1] == nt
        ):

            return prediction[0]

    if prediction.ndim == 4:

        return prediction

    raise ValueError(
        f"Unexpected rollout shape: "
        f"{prediction.shape}"
    )


def rollout(
    method,
    state,
    q_true,
    bed,
    *,
    steps,
):

    q0 = jnp.asarray(
        q_true[
            0:1
        ],
        dtype=jnp.float32,
    )

    condition = jnp.asarray(
        bed[
            None,
            ...,
            None,
        ],
        dtype=jnp.float32,
    )

    prediction = method.uniform_inference(
        state,
        q0,
        trajectory_points_num=(
            q_true.shape[0]
        ),
        steps_per_segment=steps,
        condition=condition,
        method="RK4",
    )

    return normalize_prediction(
        prediction,
        q_true.shape[0],
    )


# =====================================================================
# DIRECT VECTOR FIELD
# =====================================================================

def vector_field(
    method,
    state,
    q,
    b,
    t,
):

    result = method._model_apply(
        state.params,

        jnp.asarray(
            q[None],
            dtype=jnp.float32,
        ),

        jnp.asarray(
            [t],
            dtype=jnp.float32,
        ),

        jnp.asarray(
            b[
                None,
                ...,
                None
            ],
            dtype=jnp.float32,
        ),
    )

    return np.asarray(
        result[0],
        dtype=np.float32,
    )


# =====================================================================
# PHYSICS
# =====================================================================

def central_x(field, dx):

    return (
        np.roll(
            field,
            -1,
            axis=0,
        )
        -
        np.roll(
            field,
            1,
            axis=0,
        )
    ) / (
        2.0
        *
        dx
    )


def central_y(field, dy):

    return (
        np.roll(
            field,
            -1,
            axis=1,
        )
        -
        np.roll(
            field,
            1,
            axis=1,
        )
    ) / (
        2.0
        *
        dy
    )


def source_response(q, b, dx, dy):

    h = np.maximum(
        q[..., 0],
        1.0e-6,
    )

    hu = (
        -GRAVITY
        *
        h
        *
        central_x(
            b,
            dx,
        )
    )

    hv = (
        -GRAVITY
        *
        h
        *
        central_y(
            b,
            dy,
        )
    )

    return hu, hv


def speed(q):

    h = np.maximum(
        q[..., 0],
        1.0e-6,
    )

    u = (
        q[..., 1]
        /
        h
    )

    v = (
        q[..., 2]
        /
        h
    )

    return np.sqrt(
        u**2
        +
        v**2
    )


# =====================================================================
# METRICS
# =====================================================================

def relative_l2(truth, pred):

    numerator = np.linalg.norm(
        (
            pred
            -
            truth
        ).ravel()
    )

    denominator = np.linalg.norm(
        truth.ravel()
    )

    return float(
        numerator
        /
        max(
            denominator,
            1.0e-12,
        )
    )


def rmse(truth, pred):

    return float(
        np.sqrt(
            np.mean(
                (
                    pred
                    -
                    truth
                )
                ** 2
            )
        )
    )


def channel_metrics(truth, pred):

    return {
        "rel_l2":
            relative_l2(
                truth,
                pred,
            ),

        "rmse":
            rmse(
                truth,
                pred,
            ),

        "h_rel_l2":
            relative_l2(
                truth[..., 0],
                pred[..., 0],
            ),

        "hu_rel_l2":
            relative_l2(
                truth[..., 1],
                pred[..., 1],
            ),

        "hv_rel_l2":
            relative_l2(
                truth[..., 2],
                pred[..., 2],
            ),
    }


def error_vs_time(truth, prediction):

    values = []

    for i in range(
        truth.shape[0]
    ):

        values.append(
            relative_l2(
                truth[i],
                prediction[i],
            )
        )

    return np.asarray(
        values
    )


# =====================================================================
# HEATMAP
# =====================================================================

def heatmap(
    ax,
    field,
    *,
    title,
    extent,
    vlim=None,
):

    if vlim is None:

        vlim = max(
            float(
                np.max(
                    np.abs(
                        field
                    )
                )
            ),
            1e-12,
        )

    image = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-vlim,
        vmax=vlim,
        interpolation="bilinear",
        aspect="equal",
    )

    ax.set_title(
        title
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    return image


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    output = resolve(
        args.output_dir
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # LOAD GAUSSIAN DATA
    # -----------------------------------------------------------------

    g64 = load_case(
        args.gaussian64,
        "hill_center",
    )

    f64 = load_case(
        args.gaussian64,
        "flat",
    )

    g128 = load_case(
        args.gaussian128,
        "hill_center",
    )

    f128 = load_case(
        args.gaussian128,
        "flat",
    )

    r64 = load_case(
        args.gaussian64,
        "hill_right",
    )

    r128 = load_case(
        args.gaussian128,
        "hill_right",
    )

    # -----------------------------------------------------------------
    # LOAD OOD TWO-HILL DATA
    # -----------------------------------------------------------------

    o64 = load_case(
        args.ood64,
        "two_hills_00",
    )

    of64 = load_case(
        args.ood64,
        "flat",
    )

    o128 = load_case(
        args.ood128,
        "two_hills_00",
    )

    of128 = load_case(
        args.ood128,
        "flat",
    )

    # -----------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------

    method, state = build_zero_shot_model(
        args.checkpoint
    )

    # -----------------------------------------------------------------
    # ZERO-SHOT 128 ROLLOUTS
    # -----------------------------------------------------------------

    print(
        "Running zero-shot Gaussian hill..."
    )

    z_gaussian = rollout(
        method,
        state,
        g128["q"],
        g128["b"],
        steps=args.steps_per_segment,
    )

    print(
        "Running zero-shot Gaussian flat..."
    )

    z_flat = rollout(
        method,
        state,
        f128["q"],
        f128["b"],
        steps=args.steps_per_segment,
    )

    print(
        "Running zero-shot OOD two-hills..."
    )

    z_ood = rollout(
        method,
        state,
        o128["q"],
        o128["b"],
        steps=args.steps_per_segment,
    )

    print(
        "Running zero-shot OOD flat..."
    )

    z_ood_flat = rollout(
        method,
        state,
        of128["q"],
        of128["b"],
        steps=args.steps_per_segment,
    )

    # -----------------------------------------------------------------
    # INTERPOLATE 64 REFERENCE TO 128 GRID
    # -----------------------------------------------------------------

    q64_to_128 = interpolate_state(
        g64["q"],
        g64["x"],
        g64["y"],
        g128["x"],
        g128["y"],
    )

    flat64_to_128 = interpolate_state(
        f64["q"],
        f64["x"],
        f64["y"],
        g128["x"],
        g128["y"],
    )

    ood64_to_128 = interpolate_state(
        o64["q"],
        o64["x"],
        o64["y"],
        o128["x"],
        o128["y"],
    )

    oodflat64_to_128 = interpolate_state(
        of64["q"],
        of64["x"],
        of64["y"],
        o128["x"],
        o128["y"],
    )

    b64_to_128 = interpolate_bed(
        g64["b"],
        g64["x"],
        g64["y"],
        g128["x"],
        g128["y"],
    )

    # -----------------------------------------------------------------
    # TIME INDEX
    # -----------------------------------------------------------------

    ti = int(
        np.argmin(
            np.abs(
                g128["time"]
                -
                args.plot_time
            )
        )
    )

    tvalue = float(
        g128["time"][ti]
    )

    extent = [
        XMIN,
        XMAX,
        YMIN,
        YMAX,
    ]

    labels = [
        "PyClaw 64×64",
        "Geo-U-FNO 64→128 zero-shot",
        "PyClaw 128×128",
    ]

    # =================================================================
    # 01 — DIRECT CONDITION RESPONSE
    # =================================================================

    dx64 = 5.0 / 64.0
    dx128 = 5.0 / 128.0

    hu64, hv64 = source_response(
        r64["q"][ti],
        r64["b"],
        dx64,
        dx64,
    )

    hu64 = interp2(
        hu64,
        r64["x"],
        r64["y"],
        r128["x"],
        r128["y"],
    )

    hv64 = interp2(
        hv64,
        r64["x"],
        r64["y"],
        r128["x"],
        r128["y"],
    )

    hu128, hv128 = source_response(
        r128["q"][ti],
        r128["b"],
        dx128,
        dx128,
    )

    vf_b = vector_field(
        method,
        state,
        r128["q"][ti],
        r128["b"],
        tvalue,
    )

    vf_0 = vector_field(
        method,
        state,
        r128["q"][ti],
        np.zeros_like(
            r128["b"]
        ),
        tvalue,
    )

    delta = (
        vf_b
        -
        vf_0
    )

    hu_zero = delta[..., 1]
    hv_zero = delta[..., 2]

    hu_lim = max(
        np.max(
            np.abs(
                [
                    hu64,
                    hu_zero,
                    hu128,
                ]
            )
        ),
        1e-10,
    )

    hv_lim = max(
        np.max(
            np.abs(
                [
                    hv64,
                    hv_zero,
                    hv128,
                ]
            )
        ),
        1e-10,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            9,
        ),
    )

    hu_fields = [
        hu64,
        hu_zero,
        hu128,
    ]

    hv_fields = [
        hv64,
        hv_zero,
        hv128,
    ]

    for j in range(3):

        im = heatmap(
            axes[0, j],
            hu_fields[j],
            title=(
                labels[j]
                +
                "\nBathymetry response: hu"
            ),
            extent=extent,
            vlim=hu_lim,
        )

        fig.colorbar(
            im,
            ax=axes[0, j],
            shrink=0.8,
        )

        im = heatmap(
            axes[1, j],
            hv_fields[j],
            title=(
                labels[j]
                +
                "\nBathymetry response: hv"
            ),
            extent=extent,
            vlim=hv_lim,
        )

        fig.colorbar(
            im,
            ax=axes[1, j],
            shrink=0.8,
        )

    fig.suptitle(
        f"Direct bathymetry-condition comparison, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "01_direct_condition_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 02 — ISOLATED HILL EFFECT ERROR
    # =================================================================

    true_effect128 = (
        g128["q"]
        -
        f128["q"]
    )

    effect64 = (
        q64_to_128
        -
        flat64_to_128
    )

    zero_effect = (
        z_gaussian
        -
        z_flat
    )

    err64 = error_vs_time(
        true_effect128,
        effect64,
    )

    errzero = error_vs_time(
        true_effect128,
        zero_effect,
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.plot(
        g128["time"],
        err64,
        label=(
            "PyClaw 64→128 interpolation "
            "vs PyClaw 128"
        ),
    )

    ax.plot(
        g128["time"],
        errzero,
        label=(
            "Geo-U-FNO 64→128 zero-shot "
            "vs PyClaw 128"
        ),
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Relative terrain-effect error")
    ax.set_title("Isolated Gaussian-hill effect error")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output
        /
        "02_isolated_hill_effect_error.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 03 — GAUSSIAN TERRAIN-INDUCED SPEED
    # =================================================================

    speed64 = (
        speed(
            q64_to_128[ti]
        )
        -
        speed(
            flat64_to_128[ti]
        )
    )

    speedzero = (
        speed(
            z_gaussian[ti]
        )
        -
        speed(
            z_flat[ti]
        )
    )

    speed128 = (
        speed(
            g128["q"][ti]
        )
        -
        speed(
            f128["q"][ti]
        )
    )

    speed_lim = max(
        np.max(
            np.abs(
                [
                    speed64,
                    speedzero,
                    speed128,
                ]
            )
        ),
        1e-10,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.8,
        ),
    )

    for ax, field, label in zip(
        axes,
        [
            speed64,
            speedzero,
            speed128,
        ],
        labels,
    ):

        im = heatmap(
            ax,
            field,
            title=label,
            extent=extent,
            vlim=speed_lim,
        )

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.8,
        )

    fig.suptitle(
        f"Terrain-induced speed: Gaussian hill, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "03_terrain_induced_speed_gaussian.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 04 — UNSEEN TWO-HILL TERRAIN SPEED
    # =================================================================

    ood_speed64 = (
        speed(
            ood64_to_128[ti]
        )
        -
        speed(
            oodflat64_to_128[ti]
        )
    )

    ood_speedzero = (
        speed(
            z_ood[ti]
        )
        -
        speed(
            z_ood_flat[ti]
        )
    )

    ood_speed128 = (
        speed(
            o128["q"][ti]
        )
        -
        speed(
            of128["q"][ti]
        )
    )

    ood_lim = max(
        np.max(
            np.abs(
                [
                    ood_speed64,
                    ood_speedzero,
                    ood_speed128,
                ]
            )
        ),
        1e-10,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.8,
        ),
    )

    for ax, field, label in zip(
        axes,
        [
            ood_speed64,
            ood_speedzero,
            ood_speed128,
        ],
        labels,
    ):

        im = heatmap(
            ax,
            field,
            title=label,
            extent=extent,
            vlim=ood_lim,
        )

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.8,
        )

    fig.suptitle(
        f"Terrain-induced speed: unseen two-hill terrain, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "04_terrain_induced_speed_unseen.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 05 — h, hu, hv CROSS SECTIONS
    # =================================================================

    ix64 = int(
        np.argmin(
            np.abs(
                g64["x"]
            )
        )
    )

    iy64 = int(
        np.argmin(
            np.abs(
                g64["y"]
            )
        )
    )

    ix128 = int(
        np.argmin(
            np.abs(
                g128["x"]
            )
        )
    )

    iy128 = int(
        np.argmin(
            np.abs(
                g128["y"]
            )
        )
    )

    channel_names = [
        "h",
        "hu",
        "hv",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            15,
            8,
        ),
    )

    for c in range(3):

        axes[0, c].plot(
            g64["x"],
            g64["q"][
                ti,
                :,
                iy64,
                c,
            ],
            label="PyClaw 64×64",
        )

        axes[0, c].plot(
            g128["x"],
            z_gaussian[
                ti,
                :,
                iy128,
                c,
            ],
            label="Geo-U-FNO 64→128",
        )

        axes[0, c].plot(
            g128["x"],
            g128["q"][
                ti,
                :,
                iy128,
                c,
            ],
            label="PyClaw 128×128",
        )

        axes[0, c].set_title(
            f"{channel_names[c]} — x section"
        )

        axes[0, c].set_xlabel("x")
        axes[0, c].grid(alpha=0.25)
        axes[0, c].legend()

        axes[1, c].plot(
            g64["y"],
            g64["q"][
                ti,
                ix64,
                :,
                c,
            ],
            label="PyClaw 64×64",
        )

        axes[1, c].plot(
            g128["y"],
            z_gaussian[
                ti,
                ix128,
                :,
                c,
            ],
            label="Geo-U-FNO 64→128",
        )

        axes[1, c].plot(
            g128["y"],
            g128["q"][
                ti,
                ix128,
                :,
                c,
            ],
            label="PyClaw 128×128",
        )

        axes[1, c].set_title(
            f"{channel_names[c]} — y section"
        )

        axes[1, c].set_xlabel("y")
        axes[1, c].grid(alpha=0.25)
        axes[1, c].legend()

    fig.suptitle(
        f"Gaussian-hill cross sections, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "05_gaussian_cross_sections_h_hu_hv.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 06 — 3D BATHYMETRY RESOLUTION
    # =================================================================

    X64, Y64 = np.meshgrid(
        g64["x"],
        g64["y"],
        indexing="ij",
    )

    X128, Y128 = np.meshgrid(
        g128["x"],
        g128["y"],
        indexing="ij",
    )

    fig = plt.figure(
        figsize=(
            16,
            5,
        )
    )

    ax1 = fig.add_subplot(
        131,
        projection="3d",
    )

    ax1.plot_surface(
        X64,
        Y64,
        g64["b"],
        alpha=0.85,
    )

    ax1.set_title(
        "Bathymetry 64×64"
    )

    ax1.legend(
        handles=[
            Patch(
                label="64×64 bed"
            )
        ]
    )

    ax2 = fig.add_subplot(
        132,
        projection="3d",
    )

    ax2.plot_surface(
        X128,
        Y128,
        g128["b"],
        alpha=0.85,
    )

    ax2.set_title(
        "Bathymetry supplied to zero-shot 128"
    )

    ax2.legend(
        handles=[
            Patch(
                label="128×128 bed input"
            )
        ]
    )

    ax3 = fig.add_subplot(
        133
    )

    ax3.plot(
        g64["x"],
        g64["b"][
            :,
            iy64
        ],
        label="64×64 bed",
    )

    ax3.plot(
        g128["x"],
        g128["b"][
            :,
            iy128
        ],
        label="128×128 bed",
    )

    ax3.set_xlabel("x")
    ax3.set_ylabel("b")
    ax3.set_title(
        "Central bathymetry cross-section"
    )

    ax3.grid(alpha=0.25)
    ax3.legend()

    fig.tight_layout()

    fig.savefig(
        output
        /
        "06_gaussian_3d_bathymetry_cross_sections.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 07 — 3D FREE SURFACE
    # =================================================================

    eta64 = (
        q64_to_128[
            ti,
            ...,
            0
        ]
        +
        b64_to_128
    )

    etazero = (
        z_gaussian[
            ti,
            ...,
            0
        ]
        +
        g128["b"]
    )

    eta128 = (
        g128["q"][
            ti,
            ...,
            0
        ]
        +
        g128["b"]
    )

    eta_fields = [
        eta64,
        etazero,
        eta128,
    ]

    zmin = min(
        np.min(x)
        for x in eta_fields
    )

    zmax = max(
        np.max(x)
        for x in eta_fields
    )

    fig = plt.figure(
        figsize=(
            18,
            5,
        )
    )

    for j, (
        field,
        label,
    ) in enumerate(
        zip(
            eta_fields,
            labels,
        ),
        start=1,
    ):

        ax = fig.add_subplot(
            1,
            3,
            j,
            projection="3d",
        )

        ax.plot_surface(
            X128,
            Y128,
            field,
            alpha=0.88,
        )

        ax.set_zlim(
            zmin,
            zmax,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("η")

        ax.set_title(label)

        ax.legend(
            handles=[
                Patch(
                    label=label
                )
            ]
        )

    fig.suptitle(
        f"Free-surface comparison, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "07_gaussian_3d_free_surface.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # 08 — 3D STACKED CROSS-SECTIONS
    # =================================================================

    fig = plt.figure(
        figsize=(
            18,
            6,
        )
    )

    sections = [
        (
            "PyClaw 64×64",
            g64["x"],
            g64["q"][
                ti,
                :,
                iy64,
            ],
            0.0,
        ),
        (
            "Geo-U-FNO 64→128",
            g128["x"],
            z_gaussian[
                ti,
                :,
                iy128,
            ],
            1.0,
        ),
        (
            "PyClaw 128×128",
            g128["x"],
            g128["q"][
                ti,
                :,
                iy128,
            ],
            2.0,
        ),
    ]

    for c, name in enumerate(
        channel_names
    ):

        ax = fig.add_subplot(
            1,
            3,
            c + 1,
            projection="3d",
        )

        for (
            label,
            x,
            qsection,
            offset,
        ) in sections:

            ax.plot(
                x,
                np.full_like(
                    x,
                    offset,
                ),
                qsection[
                    ...,
                    c
                ],
                label=label,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("Solution")
        ax.set_zlabel(name)

        ax.set_yticks(
            [
                0,
                1,
                2,
            ]
        )

        ax.set_yticklabels(
            [
                "64",
                "64→128",
                "128",
            ]
        )

        ax.set_title(
            f"3D cross-section: {name}"
        )

        ax.legend(
            fontsize=8
        )

    fig.suptitle(
        f"3D Gaussian-hill state cross-sections, t={tvalue:.2f}"
    )

    fig.tight_layout()

    fig.savefig(
        output
        /
        "08_gaussian_3d_h_hu_hv_cross_sections.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =================================================================
    # FINAL NUMERICAL TABLE
    # =================================================================

    rows = []

    rows.append(
        {
            "comparison":
                "PyClaw64_interpolated_vs_PyClaw128",

            **channel_metrics(
                g128["q"],
                q64_to_128,
            ),

            "gaussian_effect_rel_l2":
                relative_l2(
                    true_effect128,
                    effect64,
                ),

            "ood_effect_rel_l2":
                relative_l2(
                    (
                        o128["q"]
                        -
                        of128["q"]
                    ),
                    (
                        ood64_to_128
                        -
                        oodflat64_to_128
                    ),
                ),
        }
    )

    rows.append(
        {
            "comparison":
                "GeoUFNO64_to_128_vs_PyClaw128",

            **channel_metrics(
                g128["q"],
                z_gaussian,
            ),

            "gaussian_effect_rel_l2":
                relative_l2(
                    true_effect128,
                    zero_effect,
                ),

            "ood_effect_rel_l2":
                relative_l2(
                    (
                        o128["q"]
                        -
                        of128["q"]
                    ),
                    (
                        z_ood
                        -
                        z_ood_flat
                    ),
                ),
        }
    )

    columns = [
        "comparison",
        "rel_l2",
        "rmse",
        "h_rel_l2",
        "hu_rel_l2",
        "hv_rel_l2",
        "gaussian_effect_rel_l2",
        "ood_effect_rel_l2",
    ]

    csv_path = (
        output
        /
        "final_resolution_metrics.csv"
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
                row
            )

    print()
    print("=" * 72)
    print("FINAL RESOLUTION COMPARISON COMPLETE")
    print("=" * 72)

    for row in rows:

        print()
        print(
            row[
                "comparison"
            ]
        )

        for key in columns[1:]:

            print(
                f"  {key:28s}: "
                f"{row[key]:.8f}"
            )

    print()
    print(
        "Results saved to:"
    )

    print(
        output
    )


if __name__ == "__main__":

    main()