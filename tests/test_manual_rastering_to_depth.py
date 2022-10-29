import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from skimage import measure
from skimage.draw import ellipsoid
import trimesh 
import cv2 
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch
from scipy.spatial import KDTree
from numpy import linalg as la
import time 
from tmp_utils import SE3, decompose_se3, blending, get_diag, dqnorm, custom_transpose_batch, get_W
import numpy as np 
from functorch import vmap, vjp, jacrev, jacfwd, hessian, jvp, grad
from einops import rearrange, reduce, repeat
import pytorch3d
import pytorch3d.utils
def test_load_mesh():
    human = trimesh.load('reconstruct/seq006/mesh.ply')
    verts, faces = human.vertices.astype(np.float32), human.faces 
    rootdir= '/home/ptthang/data_mount/KinectFusion/seq006'
    data_path = os.path.join(rootdir, "processed")
    cam_file = os.path.join(data_path, "cameras.npz")
    cam_dict = np.load(cam_file)
    print(verts.shape)
    out = cv2.decomposeProjectionMatrix(cam_dict["world_mats"][0])
    K = torch.tensor(out[0]).view(1,3,3)
    R = torch.tensor(out[1]).view(1,3,3)
    t = torch.tensor(out[2])
    t = (t[:3] / t[3])[:, 0].view(1,3)
    image_size = torch.tensor([480,640]).view(1,2)
    assert K.shape == (1,3,3), print(K.shape)
    assert R.shape == (1,3,3), print(R.shape)
    assert t.shape == (1,3), print(t, t.shape)
    # print(pytorch3d.__dict__)
    camera_torch = pytorch3d.utils.cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size)