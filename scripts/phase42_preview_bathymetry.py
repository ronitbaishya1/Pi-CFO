"""Phase 42: create and visualize Gaussian bathymetry fields.

Run from the CFO project root in either environment:

    python scripts/phase42_preview_bathymetry.py

Outputs:
    results/bathymetry_preview/gaussian_hills.png
    results/bathymetry_preview/gaussian_hills_3d.png
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.bathymetry import make_bathymetry, sample_id_parameters


def main() -> None:
    nx = ny = 32
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5

    # Cell-center coordinates, matching a finite-volume grid.
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x = x_min + (np.arange(nx) + 0.5) * dx
    y = y_min + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")

    rng = np.random.default_rng(42)

    terrains = []
    params_list = []
    for _ in range(4):
        params = sample_id_parameters(rng)
        b = make_bathymetry(X, Y, params)
        terrains.append(b)
        params_list.append(params)

    outdir = PROJECT_ROOT / "results" / "bathymetry_preview"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for i, (ax, b, params) in enumerate(zip(axes, terrains, params_list)):
        image = ax.imshow(
            b.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            aspect="equal",
        )
        ax.set_title(
            f"Hill {i+1}\n"
            f"A={params.amplitude:.3f}, "
            f"xc={params.x_center:.2f}, yc={params.y_center:.2f}, "
            f"sigma={params.sigma_x:.2f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, shrink=0.78, label="bed height b")
    fig.tight_layout()
    fig.savefig(outdir / "gaussian_hills.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 10))
    for i, b in enumerate(terrains):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.plot_surface(X, Y, b)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("b")
        ax.set_title(f"Gaussian hill {i+1}")
    fig.tight_layout()
    fig.savefig(outdir / "gaussian_hills_3d.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("=" * 60)
    print("PHASE 42 COMPLETE")
    print("=" * 60)
    print(f"dx = {dx:.8f}")
    print(f"dy = {dy:.8f}")
    for i, (b, params) in enumerate(zip(terrains, params_list)):
        print(
            f"Hill {i+1}: kind={params.kind}, "
            f"min={b.min():.6f}, max={b.max():.6f}, "
            f"A={params.amplitude:.4f}, "
            f"center=({params.x_center:.3f},{params.y_center:.3f}), "
            f"sigma={params.sigma_x:.3f}"
        )
    print(f"Plots: {outdir}")


if __name__ == "__main__":
    main()
