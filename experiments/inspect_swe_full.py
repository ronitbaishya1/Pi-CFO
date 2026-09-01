import h5py
import numpy as np

path = "data/shallow_water_full/swe_full_32_smoke.h5"

with h5py.File(path, "r") as f:

    keys = list(f.keys())

    print("Number of trajectories:", len(keys))
    print("Trajectory IDs:", keys)

    first_key = keys[0]

    data = f[first_key]["data"]

    print("\nShape:", data.shape)
    print("dtype:", data.dtype)

    q0 = data[0]
    qf = data[-1]

    print("\nINITIAL")
    print("h min/max:", q0[..., 0].min(), q0[..., 0].max())
    print("max |hu|:", np.abs(q0[..., 1]).max())
    print("max |hv|:", np.abs(q0[..., 2]).max())

    print("\nFINAL")
    print("h min/max:", qf[..., 0].min(), qf[..., 0].max())
    print("max |hu|:", np.abs(qf[..., 1]).max())
    print("max |hv|:", np.abs(qf[..., 2]).max())

    print("\nNaNs:", np.isnan(data[:]).any())