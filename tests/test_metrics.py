import numpy as np

from utils.metrics import relative_frobenius_error, rmse


def test_rmse_zero_on_identical_arrays():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    assert rmse(x, x) == 0.0


def test_relative_frobenius_error_zero_on_identical_arrays():
    x = np.ones((4, 4), dtype=np.float32)
    assert relative_frobenius_error(x, x) == 0.0
