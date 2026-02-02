import h5py

with h5py.File("P01_Isaac-Assembly-Drawer-GR1T2-Abs-v0.hdf5", "a") as f:
    g = f["/data/demo_6"]
    g.attrs["completion_time_sec"] = 81.857132021713