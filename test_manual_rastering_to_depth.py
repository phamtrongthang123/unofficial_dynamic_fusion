import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from skimage import measure
from skimage.draw import ellipsoid
import trimesh 
import cv2 
import torch
# Compose a 4x4 SE3 matrix
def compose_se3(R,t):
    M = np.identity(4)
    M[0] = np.append(R[0],t[0])
    M[1] = np.append(R[1],t[1])
    M[2] = np.append(R[2],t[2])
    return M

# Decompose M into R and t
def decompose_se3(M):
    return M[np.ix_([0,1,2],[0,1,2])], M[np.ix_([0,1,2],[3])]
def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K.astype(np.float32)


    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # convert from w2c to c2w
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose
# with open('reconstruct/seq006/mesh.ply', 'rb') as f:
human = trimesh.load('reconstruct/seq006/mesh.ply')
verts, faces = human.vertices.astype(np.float32), human.faces 
rootdir= '/home/ptthang/data_mount/KinectFusion/seq006'
data_path = os.path.join(rootdir, "processed")
cam_file = os.path.join(data_path, "cameras.npz")
cam_dict = np.load(cam_file)
print(verts.shape)
intrinsics, c2w = load_K_Rt_from_P(cam_dict["world_mats"][2])
intrinsics = torch.tensor(intrinsics.copy()).float()
print(intrinsics)
import time 
from numba import njit

t1 = time.time()
@njit
def edge_function(a,b,c):
  return (c[0]-a[0]) * (b[1]-a[1]) - (c[1] - a[1]) * (b[0] - a[0])
@njit
def get_bounding_box(points):
   return [int(x) for x in [points[:,0].min(), points[:,0].max(), points[:,1].min(), points[:,1].max()]]

poses = np.load('reconstruct/seq006/traj_gt.npz')['poses']
# print(poses.shape, poses)
R,t = decompose_se3(np.linalg.inv(poses[0]))
R,t  = torch.tensor(R), torch.tensor(t) # c2w -> world 2 cam
# print(R.shape, t.shape)
verts_t = torch.tensor(verts.copy())
faces_t = torch.tensor(faces.copy(),dtype=torch.long)
points_out = torch.zeros(480, 640)
verts_t = (intrinsics[:3, :3]@verts_t.transpose(1,0)).transpose(1,0)
verts_t = (R@verts_t.transpose(1,0) + t).transpose(1,0)
verts_before = verts_t[faces]

verts_a = verts_t / verts_t[:,2][:,None]
verts_a[:,2] = verts_t[:,2]
verts_after = verts_a[faces]
# print(verts_after[0])
def refactor(verts_after: torch.Tensor, points_out):
    # K = torch.eye(3).float()
    H, W = points_out.shape

    # N, 3, 3
    coos = verts_after

    coos = coos.int()
    xmin, _ = coos[:, :, 0].min(dim=1)
    ymin, _ = coos[:, :, 1].min(dim=1)

    xmax, _ = coos[:, :, 0].max(dim=1)
    ymax, _ = coos[:, :, 1].max(dim=1)

    # mask: shape N
    mask = (xmin < 0).logical_or(ymin < 0).logical_or(xmax > W).logical_or(ymax > H)


    # z: shape: N
    # import pdb; pdb.set_trace()
    z: torch.Tensor = (verts_after[:,0,2] + verts_after[:, 1, 2]+ verts_after[:, 2, 2]) / 3 
    

    # mask = torch.logical_and(z > 0, z < 4).logical_and(mask)
    mask = mask.logical_or(z < 0).logical_or(z > 4)
    z[mask] = float("infinity")

    n = coos.shape[0]

    #  big_points_out = torch.ones((n, H, W))
    #  tmp_mask = torch.ones(big_points_out, dtype=int) * n
    tmp_mask = torch.ones((H, W), dtype=int) * n
    index = torch.argsort(z, descending=True)

    # move back to numpy and list to reduce pytorch's overhead
    xmin = xmin[index].tolist()
    ymin = ymin[index].tolist()
    ymax = ymax[index].tolist()
    xmax = xmax[index].tolist()
    # points_out = torch.zeros((200, 640))
    index = index.cpu().numpy()
    tmp_mask = tmp_mask.cpu().numpy()

    for i, x1, y1, x2, y2 in zip(index, xmin, ymin, xmax, ymax):
        tmp_mask[y1:y2, x1:x2] = i

    z = np.concatenate([z, np.array([float("infinity")])])
    points_out = z[tmp_mask]

    points_out[np.isinf(points_out)] = 0

    return points_out
points_out = refactor(verts_after, points_out)
# points_out = try_run(verts_after, verts_before)
print("done ", time.time() - t1)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(points_out) # (h,w)
plt.colorbar(label='Distance to Camera')
plt.title('Depth image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()