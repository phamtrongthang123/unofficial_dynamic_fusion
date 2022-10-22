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
intrinsics = intrinsics.astype(np.float32)
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
print(poses.shape, poses)
R,t = decompose_se3(torch.inverse(poses[0])) # c2w -> world 2 cam
print(R.shape, t.shape)
verts = (intrinsics[:3, :3]@verts.transpose()).transpose()
verts = (R@verts.transpose() + t).transpose()
verts_before = verts[faces]

verts = verts[:,:2] / verts[:,2][:,None]
verts_after = verts[faces]
print(verts_after[0])
@njit
def try_run(verts_after, verts_before):
  points_out = np.zeros((480, 640))
  H,W = points_out.shape
#   K = np.eye(3).astype(np.float32)
  for i in range(verts_after.shape[0]):
    vv = verts_before[i]
    # print(K.dot(vv.transpose()))
    coos = verts_after[i]
    # print(coos.shape, vv[:,2].reshape(3,1))

    xmin,xmax,ymin,ymax = get_bounding_box(coos)

    if xmin < 0 or ymin < 0 or xmax > W or ymax > H:
      # print(xmin,xmax,ymin,ymax)
      continue

    for x in range(xmin,xmax,1):
      for y in range(ymin,ymax,1):
        z = (vv[0][2] + vv[1][2] + vv[2][2])/3
        # print(z)

        if z>0 and (z < points_out[y,x] or points_out[y,x]<1e-9) and z < 4000:
          points_out[y,x] = z
          # print(points_out[x,y])
  return points_out
points_out = try_run(verts_after, verts_before)
print("done ", time.time() - t1)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(points_out) # (h,w)
plt.colorbar(label='Distance to Camera')
plt.title('Depth image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()