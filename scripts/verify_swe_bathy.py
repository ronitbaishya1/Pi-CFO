"""Phases 46 and 48: verify the bathymetry solver and HDF5 dataset.

Run in `swegen` after creating the dataset:

    python scripts/verify_swe_bathy.py \
        --dataset data/shallow_water_bathy/swe_bathy_32_id.h5

This checks:
- expected shapes
- NaN/Inf
- positive depth
- eta = h+b
- ranges of h, hu, hv, b
- a lake-at-rest PyClaw simulation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This verifier is also intended to run in the lightweight `swegen`
# environment. Import bathymetry.py directly so utils/__init__.py does not
# pull in CFO-only checkpoint dependencies such as Orbax.
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from scripts.generate_swe_bathy import build_grid, run_pyclaw
from bathymetry import make_bathymetry, sample_id_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/shallow_water_bathy/swe_bathy_32_id.h5",
    )
    parser.add_argument("--lake-tolerance", type=float, default=5e-4)
    parser.add_argument("--momentum-tolerance", type=float, default=5e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = (PROJECT_ROOT / args.dataset).resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5:
        print("=" * 70)
        print("HDF5 DATASET CHECK")
        print("=" * 70)
        print(f"file: {path}")
        print(f"groups: {list(h5.keys())}")

        x = np.asarray(h5["x"])
        y = np.asarray(h5["y"])

        split_names = [
            key for key in h5.keys()
            if isinstance(h5[key], h5py.Group)
        ]

        if not split_names:
            raise RuntimeError("No split groups found.")

        for split in split_names:
            group = h5[split]
            q = np.asarray(group["q"])
            b = np.asarray(group["bathymetry"])

            if q.ndim != 5 or q.shape[-1] != 3:
                raise AssertionError(
                    f"{split}: expected q (B,T,nx,ny,3), got {q.shape}"
                )
            if b.ndim != 4 or b.shape[-1] != 1:
                raise AssertionError(
                    f"{split}: expected b (B,nx,ny,1), got {b.shape}"
                )
            if q.shape[0] != b.shape[0] or q.shape[2:4] != b.shape[1:3]:
                raise AssertionError(f"{split}: q and bathymetry shapes disagree.")
            if not np.isfinite(q).all() or not np.isfinite(b).all():
                raise AssertionError(f"{split}: NaN/Inf detected.")
            if np.min(q[..., 0]) <= 0.0:
                raise AssertionError(f"{split}: non-positive depth detected.")

            eta = q[..., 0] + b[:, None, ..., 0]

            print(f"\n{split}")
            print(f"  q shape    = {q.shape}")
            print(f"  b shape    = {b.shape}")
            print(
                f"  h range    = [{q[...,0].min():.6f}, "
                f"{q[...,0].max():.6f}]"
            )
            print(
                f"  hu range   = [{q[...,1].min():.6f}, "
                f"{q[...,1].max():.6f}]"
            )
            print(
                f"  hv range   = [{q[...,2].min():.6f}, "
                f"{q[...,2].max():.6f}]"
            )
            print(
                f"  b range    = [{b.min():.6f}, {b.max():.6f}]"
            )
            print(
                f"  eta range  = [{eta.min():.6f}, {eta.max():.6f}]"
            )

        # Plot the first training example.
        if "train" in h5:
            q = np.asarray(h5["train/q"])[0]
            b = np.asarray(h5["train/bathymetry"])[0, ..., 0]
            times = np.asarray(h5["train/time"])
            eta = q[..., 0] + b[None, ...]

            outdir = PROJECT_ROOT / "results" / "bathymetry_dataset_check"
            outdir.mkdir(parents=True, exist_ok=True)

            indices = [0, len(times)//3, 2*len(times)//3, len(times)-1]
            fig, axes = plt.subplots(3, 4, figsize=(14, 10))
            for col, ti in enumerate(indices):
                fields = [b, q[ti, ..., 0], eta[ti]]
                titles = ["Bathymetry b", "Depth h", "Free surface eta"]
                for row, (field, title) in enumerate(zip(fields, titles)):
                    im = axes[row, col].imshow(
                        field.T,
                        origin="lower",
                        extent=[x.min(), x.max(), y.min(), y.max()],
                        aspect="equal",
                    )
                    axes[row, col].set_title(f"{title}, t={times[ti]:.2f}")
                    fig.colorbar(im, ax=axes[row, col], shrink=0.75)
            fig.tight_layout()
            fig.savefig(
                outdir / "first_training_terrain_check.png",
                dpi=250,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"\ncheck plot: {outdir}")

        attrs = dict(h5.attrs)

    # Independent lake-at-rest solver check.
    nx = int(attrs.get("nx", 32))
    ny = int(attrs.get("ny", 32))
    x_min = float(attrs.get("x_min", -2.5))
    x_max = float(attrs.get("x_max", 2.5))
    y_min = float(attrs.get("y_min", -2.5))
    y_max = float(attrs.get("y_max", 2.5))
    g = float(attrs.get("gravity", 1.0))

    _, _, X, Y = build_grid(nx, ny, x_min, x_max, y_min, y_max)
    rng = np.random.default_rng(777)
    params = sample_id_parameters(rng)
    b = make_bathymetry(X, Y, params).astype(np.float32)

    eta0 = 2.0
    q_lake, _ = run_pyclaw(
        bathymetry=b,
        dam_radius=0.5,
        num_times=51,
        tfinal=1.0,
        g=g,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        eta_inside=2.0,
        eta_outside=1.0,
        min_depth=0.2,
        lake_at_rest=True,
        lake_eta=eta0,
    )

    eta_lake = q_lake[..., 0] + b[None, ...]
    max_eta_error = float(np.max(np.abs(eta_lake - eta0)))
    max_momentum = float(np.max(np.abs(q_lake[..., 1:])))

    print("\n" + "=" * 70)
    print("LAKE-AT-REST CHECK")
    print("=" * 70)
    print(f"max |eta-eta0| = {max_eta_error:.8e}")
    print(f"max |hu,hv|     = {max_momentum:.8e}")

    if max_eta_error > args.lake_tolerance:
        raise AssertionError(
            "Lake-at-rest free-surface error is larger than the requested "
            f"tolerance ({args.lake_tolerance})."
        )
    if max_momentum > args.momentum_tolerance:
        raise AssertionError(
            "Lake-at-rest momentum is larger than the requested "
            f"tolerance ({args.momentum_tolerance})."
        )

    print("\nPASS: dataset and solver checks completed.")


if __name__ == "__main__":
    main()
