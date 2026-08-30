import h5py
import numpy as np

path = "data/shallow_water/2D_rdb_NA_NA.h5"

with h5py.File(path, "r") as f:

    print("Top-level groups:")
    print(list(f.keys())[:10])

    first_key = list(f.keys())[0]

    print("\nFirst trajectory/group:")
    print(first_key)

    print("\nContents:")
    print(list(f[first_key].keys()))

    data = np.array(f[first_key]["data"])

    print("\nData shape:")
    print(data.shape)

    print("\nData dtype:")
    print(data.dtype)

    print("\nMinimum:")
    print(data.min())

    print("\nMaximum:")
    print(data.max())

    print("\nAny NaNs?")
    print(np.isnan(data).any())