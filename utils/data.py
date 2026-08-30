"""Merged data helpers (loaders/datasets/splines)."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import tensorflow as tf
from flax.jax_utils import prefetch_to_device
from jax import local_device_count
from jax.tree_util import tree_map

from .splines import linear_spline, quintic_spline_batch

# tf.config.experimental.set_visible_devices([], "GPU")


def prepare_tf_data(xs):
    num_devices = local_device_count()

    def _prepare(x):
        x = x._numpy()
        return x.reshape((num_devices, -1) + x.shape[1:])

    return tree_map(_prepare, xs)


def build_dataloader(X, t1=None, t2=None, c=None, y=None, batch_size=64, num_epochs=1, seed=0):
    """Create a `tf.data.Dataset` yielding `(X, [t1], [t2], [c])` batches.

    Args:
        X: Primary input tensor.
        t1: First optional auxiliary tensor (e.g., start time).
        t2: Second optional auxiliary tensor (e.g., end time).
        c: Optional condition tensor.
        y: Backward-compatible alias for `t1`.
    """
    if t1 is None and y is not None:
        t1 = y

    shuffle_buffer_size = 16 * batch_size
    elems = (X,) + ((t1,) if t1 is not None else ()) + ((t2,) if t2 is not None else ()) + ((c,) if c is not None else ())

    ds = tf.data.Dataset.from_tensor_slices(elems)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 16
    ds = ds.with_options(options)
    ds = ds.cache()
    ds = ds.shuffle(shuffle_buffer_size, seed=seed)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(8)
    return ds


def autoregressive_dataset(data):
    """Build one-step `(x_t, x_{t+1})` pairs from trajectory data."""
    B, T, *spatial = data.shape
    windows = sliding_window_view(data, window_shape=2, axis=1)
    x = windows[..., 0]
    y = windows[..., 1]
    x_flat = x.reshape(-1, *spatial)
    y_flat = y.reshape(-1, *spatial)
    return x_flat, y_flat


def load_partial_data(data, ratio, seed=43):
    """Randomly subsample a fraction of time snapshots from each trajectory.

    Returns:
        partial_data: np.ndarray of shape (B, K, ...)
        time_values: np.ndarray of shape (B, K), normalized to [0, 1]
    """
    data = np.asarray(data)
    if data.ndim < 2:
        raise ValueError("`data` must have shape (B, T, ...).")

    if not (0.0 < float(ratio) <= 1.0):
        raise ValueError(f"`ratio` must be in (0, 1], got {ratio}.")

    B, T = data.shape[:2]
    K = max(2, int(np.round(T * float(ratio))))
    K = min(K, T)

    rng = np.random.default_rng(seed)
    rand = rng.random((B, T))
    perm = np.argsort(rand, axis=1)
    time_indices = np.sort(perm[:, :K], axis=1)

    batch_idx = np.arange(B)[:, None]
    partial_data = data[batch_idx, time_indices]
    denom = max(T - 1, 1)
    time_values = time_indices.astype(np.float32) / float(denom)
    return partial_data, time_values


def select_data_split(splits: dict[str, np.ndarray], split: str):
    """Select one split from a split dictionary (e.g., `{"train", "eval", "test"}`)."""
    split_key = str(split).lower()
    if split_key not in splits:
        raise ValueError(f"Unsupported split='{split}'. Expected one of {tuple(splits.keys())}.")
    return splits[split_key]


__all__ = [
    "prepare_tf_data",
    "prefetch_to_device",
    "build_dataloader",
    "autoregressive_dataset",
    "load_partial_data",
    "select_data_split",
    "linear_spline",
    "quintic_spline_batch",
]
