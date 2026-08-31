"""Dataset loading helpers for supported CFO datasets."""

import h5py
import numpy as np
from scipy.io import loadmat
from typing import Callable, Tuple


# ============================================================
# LORENZ
# ============================================================

def load_Lorenz(
    file_path: str = "data/lorenz/lorenz63_T1000_dt5e-3.npz",
    train_ratio: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Load Lorenz trajectories and return normalized train/test splits."""

    data = np.load(file_path)

    traj = data["traj"]

    train_data = traj[: int(len(traj) * train_ratio)]
    test_data = traj[int(len(traj) * train_ratio):]

    train_max = np.max(train_data)
    train_min = np.min(train_data)

    scale = train_max - train_min
    shift = train_min

    train_data = (train_data - shift) / scale
    test_data = (test_data - shift) / scale

    return train_data, test_data


# ============================================================
# BURGERS
# ============================================================

def load_burgers(
    file_path: str = "data/burgers/Burger.mat",
    seed: int = 43,
) -> tuple[np.ndarray, np.ndarray]:
    """Load 1D Burgers trajectories and return randomized train/test splits."""

    data = loadmat(file_path)

    output_data = np.array(data["output"])

    training_ratio = 10 / 12

    np.random.seed(seed)

    indices = np.arange(len(output_data))
    np.random.shuffle(indices)

    training_num = int(training_ratio * len(output_data))

    train_indices = indices[:training_num]
    test_indices = indices[training_num:]

    train_data = output_data[
        train_indices,
        :,
        :-1,
    ]

    test_data = output_data[
        test_indices,
        :,
        :-1,
    ]

    return train_data, test_data


# ============================================================
# DIFFUSION-REACTION
# ============================================================

def load_diffusion_reaction(
    file_path: str = "data/diffusion_reaction/2D_diff-react_NA_NA.h5",
    training_ratio: float = 9 / 10,
    random_seed: int = 42,
    start_ind: int = 1,
    end_ind: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Load diffusion-reaction trajectories from PDEBench format."""

    data_list = []

    with h5py.File(file_path, "r") as h5_file:

        for group_name in h5_file.keys():

            data = np.array(
                h5_file[f"{group_name}/data"],
                dtype=np.float32,
            )

            data_list.append(data)

    full_dataset = np.stack(
        data_list,
        axis=0,
    )

    np.random.seed(random_seed)

    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)

    training_num = int(
        training_ratio * len(full_dataset)
    )

    train_indices = indices[:training_num]
    test_indices = indices[training_num:]

    train_data = full_dataset[
        train_indices,
        start_ind:end_ind,
    ]

    test_data = full_dataset[
        test_indices,
        start_ind:end_ind,
    ]

    return train_data, test_data


# ============================================================
# ORIGINAL CFO / PDEBENCH SHALLOW WATER
# ============================================================

def load_shallow_water_pdebench(
    file_path: str = "data/shallow_water/2D_rdb_NA_NA.h5",
    training_ratio: float = 9 / 10,
    random_seed: int = 42,
    start_ind: int = 1,
    end_ind: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Load original shallow-water trajectories from PDEBench format."""

    dataset = []

    with h5py.File(file_path, "r") as f:

        keys = list(f.keys())

        for key in keys:

            data = np.array(
                f[key]["data"],
                dtype=np.float32,
            )

            dataset.append(data)

    dataset = np.array(
        dataset,
        dtype=np.float32,
    )

    np.random.seed(random_seed)

    n_samples = len(dataset)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_size = int(
        training_ratio * n_samples
    )

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = dataset[
        train_indices,
        start_ind:end_ind,
    ]

    test_data = dataset[
        test_indices,
        start_ind:end_ind,
    ]

    return train_data, test_data


# ============================================================
# OUR NEW SMALL FULL-STATE SWE DATASET
# ============================================================

def load_shallow_water_full_small(
    file_path: str = "data/shallow_water_full/swe_full_32_small.h5",
    training_ratio: float = 2 / 3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load our full-state shallow-water dataset.

    Expected shape:

        (N_trajectories, N_time, Nx, Ny, 3)

    Channels:

        0 -> h
        1 -> hu
        2 -> hv

    For 30 trajectories we expect approximately:

        20 training
        10 held-out

    load_dataset_splits() will later divide the
    10 held-out trajectories into:

        5 validation
        5 test
    """

    print(
        f"Loading full-state SWE dataset from:\n"
        f"{file_path}"
    )

    dataset = []

    with h5py.File(file_path, "r") as f:

        keys = sorted(f.keys())

        for key in keys:

            data = np.asarray(
                f[key]["data"],
                dtype=np.float32,
            )

            dataset.append(data)

    dataset = np.stack(
        dataset,
        axis=0,
    )

    print(
        "Full-state SWE dataset shape:",
        dataset.shape,
    )

    # --------------------------------------------------------
    # Randomize trajectory ordering
    # --------------------------------------------------------

    rng = np.random.default_rng(
        random_seed
    )

    indices = np.arange(
        len(dataset)
    )

    rng.shuffle(indices)

    dataset = dataset[indices]

    # --------------------------------------------------------
    # Train / held-out split
    # --------------------------------------------------------

    n_total = len(dataset)

    n_train = int(
        training_ratio * n_total
    )

    train_data = dataset[:n_train]

    heldout_data = dataset[n_train:]

    print(
        "Training trajectories:",
        train_data.shape,
    )

    print(
        "Held-out trajectories:",
        heldout_data.shape,
    )

    return train_data, heldout_data


# ============================================================
# COMMON DATASET SPLITTING FUNCTION
# ============================================================

def load_dataset_splits(
    dataset: str,
    file_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Load dataset and return standardized:

        train
        eval
        test

    splits.
    """

    key = dataset.lower()

    loaders: dict[
        str,
        Callable[..., Tuple[np.ndarray, np.ndarray]]
    ] = {

        "lorenz":
            load_Lorenz,

        "burgers":
            load_burgers,

        "dr":
            load_diffusion_reaction,

        "swe":
            load_shallow_water_pdebench,

        "swe_full_small":
            load_shallow_water_full_small,
    }

    if key not in loaders:

        raise ValueError(
            f"Unsupported dataset='{dataset}'. "
            f"Supported: {sorted(loaders)}"
        )

    fn = loaders[key]

    # --------------------------------------------------------
    # Custom path if provided
    # --------------------------------------------------------

    if file_path:

        train_data, heldout_data = fn(
            file_path=file_path
        )

    else:

        train_data, heldout_data = fn()

    # --------------------------------------------------------
    # Split held-out trajectories into validation and test
    # --------------------------------------------------------

    half = len(heldout_data) // 2

    eval_data = heldout_data[:half]

    test_data = heldout_data[half:]

    return {

        "train":
            train_data,

        "eval":
            eval_data,

        "test":
            test_data,
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "load_Lorenz",
    "load_burgers",
    "load_diffusion_reaction",
    "load_shallow_water_pdebench",
    "load_shallow_water_full_small",
    "load_dataset_splits",
]