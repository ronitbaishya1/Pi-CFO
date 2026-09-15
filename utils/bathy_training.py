"""Utilities for bathymetry-conditioned CFO training."""

from __future__ import annotations

import numpy as np

from utils.data import (
    build_dataloader,
    linear_spline,
    quintic_spline_batch,
)


def broadcast_time_for_trajectories(
    time: np.ndarray,
    num_trajectories: int,
    num_snapshots: int,
    *,
    dtype=np.float32,
) -> np.ndarray:
    """Return time with shape (B, T)."""

    time = np.asarray(
        time,
        dtype=dtype,
    )

    if time.ndim == 1:

        if time.shape[0] != num_snapshots:

            raise ValueError(
                f"1D time must have length "
                f"{num_snapshots}, "
                f"got {time.shape}."
            )

        return np.broadcast_to(
            time[None, :],
            (
                num_trajectories,
                num_snapshots,
            ),
        ).copy()

    if time.shape != (
        num_trajectories,
        num_snapshots,
    ):

        raise ValueError(
            "time must have shape (T,) or (B,T); "
            f"expected ({num_snapshots},) or "
            f"({num_trajectories},{num_snapshots}), "
            f"got {time.shape}."
        )

    return time.copy()


def repeat_bathymetry_for_spline_intervals(
    bathymetry: np.ndarray,
    num_snapshots: int,
) -> np.ndarray:
    """
    Repeat each trajectory's terrain once for every spline interval.

    A trajectory with T snapshots creates T-1 spline intervals.

    Therefore:

        trajectory 0:
            b0, b0, ..., b0

        trajectory 1:
            b1, b1, ..., b1

    Each bathymetry is repeated T-1 times.
    """

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float32,
    )

    if bathymetry.ndim not in (
        3,
        4,
    ):

        raise ValueError(
            "bathymetry must have shape "
            "(B,H,W) or (B,H,W,1); "
            f"got {bathymetry.shape}."
        )

    if num_snapshots < 2:

        raise ValueError(
            "Need at least two snapshots "
            "to create spline intervals."
        )

    intervals_per_trajectory = (
        int(num_snapshots)
        -
        1
    )

    return np.repeat(
        bathymetry,
        intervals_per_trajectory,
        axis=0,
    )


def build_conditioned_spline_dataloader(
    *,
    train_data: np.ndarray,
    bathymetry: np.ndarray,
    time: np.ndarray,
    spline_type: str,
    batch_size: int,
    spline_batch_size: int,
    epochs: int,
    seed: int,
):
    """
    Build a CFO dataloader in which every temporal spline interval
    receives the bathymetry belonging to its original trajectory.
    """

    train_data = np.asarray(
        train_data,
        dtype=np.float32,
    )

    bathymetry = np.asarray(
        bathymetry,
        dtype=np.float32,
    )

    if train_data.ndim < 3:

        raise ValueError(
            "train_data must have shape "
            "(B,T,...); "
            f"got {train_data.shape}."
        )

    num_trajectories = (
        train_data.shape[0]
    )

    num_snapshots = (
        train_data.shape[1]
    )

    if (
        bathymetry.shape[0]
        !=
        num_trajectories
    ):

        raise ValueError(
            "Number of bathymetry fields must "
            "match number of trajectories: "
            f"{bathymetry.shape[0]} != "
            f"{num_trajectories}."
        )

    time = broadcast_time_for_trajectories(
        time,
        num_trajectories,
        num_snapshots,
        dtype=train_data.dtype,
    )

    if spline_type == "linear":

        (
            spline_coef,
            start_time,
            end_time,
        ) = linear_spline(
            train_data,
            time=time,
        )

    elif spline_type == "quintic":

        (
            spline_coef,
            start_time,
            end_time,
        ) = quintic_spline_batch(
            train_data,
            time=time,
            batch_size=spline_batch_size,
        )

    else:

        raise ValueError(
            "spline_type must be "
            "'linear' or 'quintic', "
            f"got {spline_type!r}."
        )

    condition = (
        repeat_bathymetry_for_spline_intervals(
            bathymetry,
            num_snapshots,
        )
    )

    expected_intervals = (
        num_trajectories
        *
        (
            num_snapshots
            -
            1
        )
    )

    if (
        len(spline_coef)
        !=
        expected_intervals
    ):

        raise RuntimeError(
            "Unexpected number of spline intervals: "
            f"expected {expected_intervals}, "
            f"got {len(spline_coef)}."
        )

    if not (
        len(spline_coef)
        ==
        len(start_time)
        ==
        len(end_time)
        ==
        len(condition)
    ):

        raise RuntimeError(
            "Spline arrays and bathymetry "
            "conditions are misaligned: "
            f"coef={len(spline_coef)}, "
            f"start={len(start_time)}, "
            f"end={len(end_time)}, "
            f"condition={len(condition)}."
        )

    return build_dataloader(
        spline_coef,
        t1=start_time,
        t2=end_time,
        c=condition,
        batch_size=batch_size,
        num_epochs=epochs,
        seed=seed,
    )


__all__ = [
    "broadcast_time_for_trajectories",
    "repeat_bathymetry_for_spline_intervals",
    "build_conditioned_spline_dataloader",
]