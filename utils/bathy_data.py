"""Phase 49: HDF5 reader for bathymetry-conditioned shallow-water data."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _read_split(group: h5py.Group) -> dict[str, np.ndarray]:
    return {
        "q": np.asarray(group["q"], dtype=np.float32),
        "bathymetry": np.asarray(group["bathymetry"], dtype=np.float32),
        "time": np.asarray(group["time"], dtype=np.float32),
        "terrain_parameters": (
            np.asarray(group["terrain_parameters"], dtype=np.float32)
            if "terrain_parameters" in group
            else np.empty((0, 0), dtype=np.float32)
        ),
        "dam_radius": (
            np.asarray(group["dam_radius"], dtype=np.float32)
            if "dam_radius" in group
            else np.empty((0,), dtype=np.float32)
        ),
    }


def load_bathymetry_dataset(file_path: str | Path) -> dict:
    """Load all split groups from one bathymetry HDF5 file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    output: dict = {}
    with h5py.File(path, "r") as h5:
        output["x"] = np.asarray(h5["x"], dtype=np.float32)
        output["y"] = np.asarray(h5["y"], dtype=np.float32)
        output["attrs"] = dict(h5.attrs)

        for key in ("train", "eval", "test_id", "test_ood"):
            if key in h5:
                output[key] = _read_split(h5[key])

    return output


def require_split(dataset: dict, split: str) -> dict[str, np.ndarray]:
    if split not in dataset:
        raise KeyError(
            f"Split '{split}' not present. Available keys: {list(dataset.keys())}"
        )
    return dataset[split]


__all__ = ["load_bathymetry_dataset", "require_split"]
