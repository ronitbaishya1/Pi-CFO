"""Bathymetry generators for the 2D shallow-water CFO project.

This module is NumPy-only so it can be imported from both environments:
- `swegen`: Clawpack/PyClaw data generation
- `cfo`: JAX/CFO training and evaluation

Conventions
-----------
b(x, y) >= 0 means the bed rises upward into the water column.
The free-surface elevation is eta = h + b.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TerrainKind = Literal["gaussian", "double_gaussian", "ridge", "narrow_gaussian"]


@dataclass(frozen=True)
class TerrainParameters:
    kind: str
    amplitude: float
    x_center: float
    y_center: float
    sigma_x: float
    sigma_y: float
    second_amplitude: float = 0.0
    second_x_center: float = 0.0
    second_y_center: float = 0.0
    second_sigma_x: float = 0.0
    second_sigma_y: float = 0.0


def gaussian_hill(
    x: np.ndarray,
    y: np.ndarray,
    *,
    amplitude: float,
    x_center: float,
    y_center: float,
    sigma_x: float,
    sigma_y: float | None = None,
) -> np.ndarray:
    """Return a smooth Gaussian hill b(x,y)."""
    if sigma_y is None:
        sigma_y = sigma_x
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("sigma_x and sigma_y must be positive.")
    exponent = -0.5 * (
        ((x - x_center) / sigma_x) ** 2
        + ((y - y_center) / sigma_y) ** 2
    )
    return amplitude * np.exp(exponent)


def double_gaussian_hill(
    x: np.ndarray,
    y: np.ndarray,
    *,
    amplitude_1: float,
    x_center_1: float,
    y_center_1: float,
    sigma_1: float,
    amplitude_2: float,
    x_center_2: float,
    y_center_2: float,
    sigma_2: float,
) -> np.ndarray:
    """Return two superposed Gaussian hills."""
    return (
        gaussian_hill(
            x,
            y,
            amplitude=amplitude_1,
            x_center=x_center_1,
            y_center=y_center_1,
            sigma_x=sigma_1,
        )
        + gaussian_hill(
            x,
            y,
            amplitude=amplitude_2,
            x_center=x_center_2,
            y_center=y_center_2,
            sigma_x=sigma_2,
        )
    )


def gaussian_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    amplitude: float,
    center: float,
    sigma: float,
    angle_radians: float,
) -> np.ndarray:
    """Return a long Gaussian ridge, rotated in the x-y plane."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    # Coordinate normal to the ridge.
    normal = np.cos(angle_radians) * x + np.sin(angle_radians) * y
    return amplitude * np.exp(-0.5 * ((normal - center) / sigma) ** 2)


def make_bathymetry(
    x: np.ndarray,
    y: np.ndarray,
    params: TerrainParameters,
) -> np.ndarray:
    """Create one terrain field from TerrainParameters."""
    if params.kind == "gaussian":
        return gaussian_hill(
            x,
            y,
            amplitude=params.amplitude,
            x_center=params.x_center,
            y_center=params.y_center,
            sigma_x=params.sigma_x,
            sigma_y=params.sigma_y,
        )

    if params.kind == "narrow_gaussian":
        return gaussian_hill(
            x,
            y,
            amplitude=params.amplitude,
            x_center=params.x_center,
            y_center=params.y_center,
            sigma_x=params.sigma_x,
            sigma_y=params.sigma_y,
        )

    if params.kind == "double_gaussian":
        return (
            gaussian_hill(
                x,
                y,
                amplitude=params.amplitude,
                x_center=params.x_center,
                y_center=params.y_center,
                sigma_x=params.sigma_x,
                sigma_y=params.sigma_y,
            )
            + gaussian_hill(
                x,
                y,
                amplitude=params.second_amplitude,
                x_center=params.second_x_center,
                y_center=params.second_y_center,
                sigma_x=params.second_sigma_x,
                sigma_y=params.second_sigma_y,
            )
        )

    if params.kind == "ridge":
        # For a ridge, x_center stores the normal-coordinate center and
        # y_center stores the angle in radians.
        return gaussian_ridge(
            x,
            y,
            amplitude=params.amplitude,
            center=params.x_center,
            sigma=params.sigma_x,
            angle_radians=params.y_center,
        )

    raise ValueError(f"Unsupported terrain kind: {params.kind}")


def sample_id_parameters(
    rng: np.random.Generator,
) -> TerrainParameters:
    """Training / in-distribution terrain: one moderate Gaussian hill."""
    sigma = float(rng.uniform(0.35, 0.70))
    return TerrainParameters(
        kind="gaussian",
        amplitude=float(rng.uniform(0.05, 0.25)),
        x_center=float(rng.uniform(-1.0, 1.0)),
        y_center=float(rng.uniform(-1.0, 1.0)),
        sigma_x=sigma,
        sigma_y=sigma,
    )


def sample_ood_parameters(
    rng: np.random.Generator,
    index: int,
) -> TerrainParameters:
    """Unseen terrain families for OOD testing.

    The training set uses only one approximately circular Gaussian hill.
    OOD examples cycle through:
      0: two hills
      1: a ridge
      2: a narrow/taller Gaussian hill
    """
    family = index % 3

    if family == 0:
        s1 = float(rng.uniform(0.25, 0.40))
        s2 = float(rng.uniform(0.25, 0.40))
        return TerrainParameters(
            kind="double_gaussian",
            amplitude=float(rng.uniform(0.08, 0.16)),
            x_center=float(rng.uniform(-1.2, -0.2)),
            y_center=float(rng.uniform(-0.8, 0.8)),
            sigma_x=s1,
            sigma_y=s1,
            second_amplitude=float(rng.uniform(0.08, 0.16)),
            second_x_center=float(rng.uniform(0.2, 1.2)),
            second_y_center=float(rng.uniform(-0.8, 0.8)),
            second_sigma_x=s2,
            second_sigma_y=s2,
        )

    if family == 1:
        return TerrainParameters(
            kind="ridge",
            amplitude=float(rng.uniform(0.10, 0.25)),
            # For ridge: x_center=center along normal, y_center=angle.
            x_center=float(rng.uniform(-0.5, 0.5)),
            y_center=float(rng.uniform(0.0, np.pi)),
            sigma_x=float(rng.uniform(0.20, 0.35)),
            sigma_y=1.0,
        )

    return TerrainParameters(
        kind="narrow_gaussian",
        amplitude=float(rng.uniform(0.22, 0.35)),
        x_center=float(rng.uniform(-1.2, 1.2)),
        y_center=float(rng.uniform(-1.2, 1.2)),
        sigma_x=float(rng.uniform(0.18, 0.30)),
        sigma_y=float(rng.uniform(0.18, 0.30)),
    )


def parameter_vector(params: TerrainParameters) -> np.ndarray:
    """Numeric representation saved in HDF5 for reproducibility.

    Columns:
    [kind_id, A, xc, yc, sx, sy, A2, xc2, yc2, sx2, sy2]
    """
    kind_ids = {
        "gaussian": 0.0,
        "double_gaussian": 1.0,
        "ridge": 2.0,
        "narrow_gaussian": 3.0,
    }
    return np.asarray(
        [
            kind_ids[params.kind],
            params.amplitude,
            params.x_center,
            params.y_center,
            params.sigma_x,
            params.sigma_y,
            params.second_amplitude,
            params.second_x_center,
            params.second_y_center,
            params.second_sigma_x,
            params.second_sigma_y,
        ],
        dtype=np.float32,
    )
