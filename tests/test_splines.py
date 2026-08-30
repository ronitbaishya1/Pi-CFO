import numpy as np

from utils.data import linear_spline, quintic_spline_batch


def test_linear_spline_shapes():
    x = np.random.randn(4, 6, 3).astype(np.float32)
    coefs, idx = linear_spline(x)
    assert coefs.shape == (4 * 5, 2, 3)
    assert idx.shape == (4 * 5,)


def test_quintic_spline_batch_shapes():
    x = np.random.randn(2, 8, 1).astype(np.float32)
    coefs, idx = quintic_spline_batch(x, batch_size=2)
    assert coefs.shape == (2 * 7, 6, 1)
    assert idx.shape == (2 * 7,)
