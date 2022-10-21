import numpy as np 
import os 
from pathlib import Path 
import shutil
seq006 = '/home/ptthang/data_mount/KinectFusion/seq006'
outdir = Path(seq006)/'processed'
intrinsic_txt = os.path.join(seq006, 'intrinsics.txt')
depth_original = Path(seq006)/'depth'
color_original = Path(seq006)/'color'
def get_calib(path:str) -> np.ndarray:
    strs = open(path).read().split()
    return np.array([[strs[0], strs[1], strs[2], strs[3]], [strs[4], strs[5], strs[6], strs[7]],
    [strs[8], strs[9], strs[10], strs[11]],[strs[12], strs[13], strs[14], strs[15]]], dtype = np.float32)

out_rgb_dir = outdir/"rgb"
if not os.path.exists(out_rgb_dir):
    os.makedirs(out_rgb_dir)
out_dep_dir = outdir/"depth"
if not os.path.exists(out_dep_dir):
    os.makedirs(out_dep_dir)
poses = [] 
for d in depth_original.glob("*"):
    fname =d.name
    shutil.copyfile(d, os.path.join(out_dep_dir, fname))
for c in color_original.glob("*"):
    fname = c.name 
    shutil.copyfile(c, out_rgb_dir/fname)
    poses.append(np.eye(4))


np.savez(os.path.join(outdir, "raw_poses.npz"), c2w_mats=poses)
camera_dict = np.load(os.path.join(outdir, "raw_poses.npz"))
K = get_calib(intrinsic_txt)[:3,:3]
poses = camera_dict["c2w_mats"]
print(poses.shape)
P_mats = []
for c2w in poses:
    w2c = np.linalg.inv(c2w)
    P = K @ w2c[:3, :]
    P_mats += [P]
print(np.array(P_mats).shape)
np.savez(os.path.join(outdir, "cameras.npz"), world_mats=P_mats)