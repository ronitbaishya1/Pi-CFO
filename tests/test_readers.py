import numpy as np

import utils.readers as readers


def test_load_lorenz_normalization_and_split(monkeypatch):
    traj = np.linspace(-3.0, 7.0, num=10 * 6 * 3, dtype=np.float32).reshape(10, 6, 3)

    monkeypatch.setattr(readers.np, "load", lambda _: {"traj": traj})
    train, test = readers.load_Lorenz(file_path="dummy.npz", train_ratio=0.8)

    assert train.shape == (8, 6, 3)
    assert test.shape == (2, 6, 3)
    assert np.isclose(train.min(), 0.0)
    assert np.isclose(train.max(), 1.0)


def test_load_burgers_split(monkeypatch):
    output = np.random.RandomState(0).randn(12, 101, 101).astype(np.float32)

    monkeypatch.setattr(readers, "loadmat", lambda _: {"output": output})
    train, test = readers.load_burgers(file_path="dummy.mat", seed=0)

    assert train.shape[0] == 10
    assert test.shape[0] == 2
    assert train.shape[1:] == (101, 100)


def test_load_dataset_splits_eval_test_halves(monkeypatch):
    train = np.zeros((9, 4, 3), dtype=np.float32)
    heldout = np.zeros((6, 4, 3), dtype=np.float32)

    monkeypatch.setattr(readers, "load_Lorenz", lambda *args, **kwargs: (train, heldout))
    splits = readers.load_dataset_splits("lorenz")

    assert splits["train"].shape[0] == 9
    assert splits["eval"].shape[0] == 3
    assert splits["test"].shape[0] == 3


def test_load_dataset_aliases(monkeypatch):
    train = np.zeros((8, 4, 2), dtype=np.float32)
    heldout = np.zeros((4, 4, 2), dtype=np.float32)

    monkeypatch.setattr(readers, "load_diffusion_reaction", lambda *args, **kwargs: (train, heldout))
    monkeypatch.setattr(readers, "load_shallow_water_pdebench", lambda *args, **kwargs: (train, heldout))

    splits_dr = readers.load_dataset_splits("dr")
    splits_swe = readers.load_dataset_splits("swe")

    assert splits_dr["train"].shape[0] == 8
    assert splits_swe["train"].shape[0] == 8
