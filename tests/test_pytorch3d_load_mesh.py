import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
import torch
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch
import time 
import numpy as np 
import pytorch3d
import pytorch3d.utils
from pytorch3d.io import IO, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)




def test_runtime_and_sanity_load_mesh():
    # running on cpu is much faster than gpu, the bottle neck is rasterizing and loading (and convert to gpu)
    # but rasterize on gpu only cost a little bit more in first run. So if we don't want to convert too much, it is not that costly. And from the second run, gpu perform much than cpu because pytorch has the memory slot.
    device=torch.device("cuda")
    print("FIRST TIME =====")
    t1 = time.time()
    # human = IO().load_mesh("reconstruct/seq006/mesh.ply", device=device)
    # verts, faces = human.verts_packed().float(), human.faces_packed().float()
    verts, faces = load_ply('reconstruct/seq006/mesh.ply')
    verts = verts.float().to(device)
    faces = faces.to(device)
    print(verts.shape)
    t2 = time.time()
    print("Load mesh cost: ", t2-t1)
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
    image_size_t = torch.tensor(image_size).view(1,2)
    assert K.shape == (1,3,3), print(K.shape)
    assert R.shape == (1,3,3), print(R.shape)
    assert t.shape == (1,3), print(t, t.shape)

    camera_torch = pytorch3d.utils.cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size_t)
    t3 = time.time()
    print("Make camera cost: ", t3-t2)
    mesh = Meshes(verts=[verts], faces=[faces])
    t4 = time.time()
    print("Init mesh cost: ", t4-t3)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
        bin_size=None,
    )
    t5 = time.time()
    print("create raster setting: ", t5-t4)
    mesh_raster = MeshRasterizer(cameras=camera_torch, raster_settings=raster_settings).to(device)
    t6 = time.time()
    print("create mesh raster: ", t6-t5)
    raster_result = mesh_raster(mesh)
    t7 = time.time()
    print("Rasterize cost: ", t7-t6)
    del raster_result, mesh_raster, raster_settings, mesh, verts, faces, camera_torch
    print("SECOND TIME =====")
    t1 = time.time()
    # human = IO().load_mesh("reconstruct/seq006/mesh.ply", device=device)
    # verts, faces = human.verts_packed().float(), human.faces_packed().float()
    verts, faces = load_ply('reconstruct/seq006/mesh.ply')
    verts = verts.float().to(device)
    faces = faces.to(device)
    print(verts.shape)
    t2 = time.time()
    print("Load mesh cost: ", t2-t1)
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
    image_size_t = torch.tensor(image_size).view(1,2)
    assert K.shape == (1,3,3), print(K.shape)
    assert R.shape == (1,3,3), print(R.shape)
    assert t.shape == (1,3), print(t, t.shape)

    camera_torch = pytorch3d.utils.cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=K, image_size=image_size_t)
    t3 = time.time()
    print("Make camera cost: ", t3-t2)
    mesh = Meshes(verts=[verts], faces=[faces])
    t4 = time.time()
    print("Init mesh cost: ", t4-t3)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
        bin_size=None,
    )
    t5 = time.time()
    print("create raster setting: ", t5-t4)
    mesh_raster = MeshRasterizer(cameras=camera_torch, raster_settings=raster_settings).to(device)
    t6 = time.time()
    print("create mesh raster: ", t6-t5)
    raster_result = mesh_raster(mesh)
    t7 = time.time()
    print("Rasterize cost: ", t7-t6)
    assert raster_result.pix_to_face.shape == (1,480,640,1), raster_result.pix_to_face.shape # (N, image_size, image_size, faces_per_pixel)
    assert raster_result.zbuf.shape == (1,480,640,1) # (N, image_size, image_size, faces_per_pixel)
    assert raster_result.bary_coords.shape == (1,480,640,1,3) # (N, image_size, image_size, faces_per_pixel, 3)
    assert raster_result.dists.shape == (1,480,640,1) #  (N, image_size, image_size, faces_per_pixel)

    # For qualitative testing, uncomment the lines below
    depth_map = raster_result.zbuf[0].view(image_size).detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(depth_map) # (h,w)
    # plt.colorbar(label='Distance to Camera')
    # plt.title('Depth image')
    # plt.xlabel('X Pixel')
    # plt.ylabel('Y Pixel')
    # plt.show()
    a = 0

def test_sanity_pytorch():
    device = "cuda"
    t1 = time.time()
    a = torch.ones((17000, 3)) 
    a = a.to(device)
    t2 = time.time()
    print("convert time: ", t2 - t1) # 
    t1 = time.time()
    a = torch.ones((17000, 3)) 
    a = a.to(device)
    t2 = time.time()
    print("convert time: ", t2 - t1) # 


