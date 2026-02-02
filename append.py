import h5py

src_path = "P03_Isaac-Assembly-Desk-GR1T2-Abs-v0.hdf5"
dst_path = "P02_Isaac-Assembly-Desk-GR1T2-Abs-v0.hdf5"

def copy_group_overwrite(src_grp, dst_grp):
    # copy attrs
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v

    for name, obj in src_grp.items():
        if name in dst_grp:
            del dst_grp[name]

        if isinstance(obj, h5py.Group):
            new_dst = dst_grp.create_group(name)
            copy_group_overwrite(obj, new_dst)
        else:
            # dataset (or other object)
            src_grp.copy(obj, dst_grp, name=name)

with h5py.File(src_path, "r+") as src, h5py.File(dst_path, "a") as dst:
    dst.require_group("/data")

    # Ensure destination demo_7 is clean
    if "/data/demo_7" in dst:
        del dst["/data/demo_7"]

    # Create empty demo_7 group then copy contents safely
    dst_demo = dst["/data"].create_group("demo_7")
    src_demo = src["/data/demo_1"]
    copy_group_overwrite(src_demo, dst_demo)

    # Delete from source only after successful copy
    if "/data/demo_1" in src:
        del src["/data/demo_1"]




