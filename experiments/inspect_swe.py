import h5py

path = "data/shallow_water/2D_rdb_NA_NA.h5"

with h5py.File(path, "r") as f:
    keys = list(f.keys())

    print("Number of top-level groups:", len(keys))
    print("First 10 groups:", keys[:10])

    first_key = keys[0]

    print("\nFirst group:", first_key)
    print("Contents:", list(f[first_key].keys()))

    data = f[first_key]["data"]

    print("\nDataset shape:", data.shape)
    print("Dataset dtype:", data.dtype)
    print("First frame shape:", data[0].shape)