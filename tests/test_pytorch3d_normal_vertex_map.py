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
from pytorch3d.io import IO, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
from pytorch3d.ops import interpolate_face_attributes

def render_depth(R,t,K, verts, faces, raster_settings, image_size, device):
    assert K.shape == (1,3,3), print(K.shape)
    assert R.shape == (1,3,3), print(R.shape)
    assert t.shape == (1,3), print(t, t.shape)

    mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    image_size_t = torch.tensor(image_size).view(1,2)
    camera_torch = pytorch3d.utils.cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size_t)
    mesh_raster = MeshRasterizer(cameras=camera_torch, raster_settings=raster_settings).to(device)
    fragments = mesh_raster(mesh)
    depth_map = fragments.zbuf[0].view(image_size).detach().cpu().numpy()
    vertex_normals = mesh.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    # normal map don't need interpolate bary coord
    normal_map = interpolate_face_attributes(fragments.pix_to_face, ones, faces_normals).view(image_size[0], image_size[1], 3)
    # vertex needs it though
    vertex_map = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, verts[faces]).view(image_size[0], image_size[1], 3)
    return depth_map,normal_map, vertex_map

def test_render_depth():
    # run under cpu for stability
    device=torch.device("cpu")
    verts, faces = load_ply('reconstruct/seq006/mesh.ply')
    verts = verts.float().to(device)
    faces = faces.to(device)
    rootdir= '/home/ptthang/data_mount/KinectFusion/seq006'
    data_path = os.path.join(rootdir, "processed")
    cam_file = os.path.join(data_path, "cameras.npz")
    cam_dict = np.load(cam_file)
    out = cv2.decomposeProjectionMatrix(cam_dict["world_mats"][0])
    K = torch.tensor(out[0]).view(1,3,3).float()
    R = torch.tensor(out[1]).view(1,3,3).float()
    t = torch.tensor(out[2])
    t = (t[:3] / t[3])[:, 0].view(1,3).float()
    image_size = [480, 640]
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
        bin_size=None,
    )
    # For qualitative testing, uncomment the lines below
    depth_map, normal_map, vertex_map = render_depth(R,t,K,verts, faces, raster_settings, image_size, device)
    assert depth_map.shape == tuple(image_size)
    assert normal_map.shape == (480, 640, 3)
    assert vertex_map.shape == (480, 640, 3)
    # for qualitative test, please uncomment the lines below
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(normal_map) # (h,w)
    # plt.colorbar(label='Distance to Camera')
    # plt.title('normal_map image')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()
    # plt.figure()
    # plt.imshow(vertex_map) # (h,w)
    # plt.colorbar(label='Distance to Camera')
    # plt.title('vertex_map image')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()
