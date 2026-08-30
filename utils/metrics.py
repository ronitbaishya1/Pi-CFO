"""Error metrics used in training and evaluation."""

import numpy as np


def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """Relative Frobenius norm error between A and B."""
    err = np.linalg.norm(A - B)
    ref = np.linalg.norm(A)
    return float(err / ref)


def relative_L2_error(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Mean and max relative RMSE over spatial dims."""
    axes = tuple(range(1, gt.ndim))
    err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
    den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
    rel = err / den
    return np.mean(rel, axis=0)


def relative_L2_error_all_steps(
    gt: np.ndarray,
    pred: np.ndarray,
    eps: float = 1e-8,
    keep_channel: bool = False,
) -> np.ndarray:
    """Per-step mean relative RMSE (optionally per channel)."""
    if keep_channel:
        axes = tuple(range(2, gt.ndim - 1))
    else:
        axes = tuple(range(2, gt.ndim))
    err = np.sqrt(np.mean((gt - pred) ** 2, axis=axes))
    den = np.sqrt(np.mean(gt**2, axis=axes)) + eps
    return np.mean(err / den, axis=0)


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    error_images = gt - pred
    root_mean_square_error = np.sqrt(np.mean(np.square(error_images)))
    return float(root_mean_square_error)
