from pathlib import Path 
from tmp_utils import render_depth
import os 
import cv2 
import torch
from pytorch3d.io import IO, load_ply
import numpy as np 
image_size = [480,640]
device = 'cuda'
import matplotlib.pyplot as plt
vis_dir = 'saved_vis'
os.makedirs(vis_dir, exist_ok=True)
root = Path('./reconstruct/seq006/warped')
lis = sorted(root.glob("*.ply"), key=lambda x: int(str(x.name).split('_')[0].split("sh")[1]))
print(lis)
for i,p in enumerate(lis):
    verts, faces = load_ply(str(p))
    verts = verts.float().to(device)
    faces = faces.to(device)
    rootdir= '/home/ptthang/data_mount/KinectFusion/seq006'
    data_path = os.path.join(rootdir, "processed")
    cam_file = os.path.join(data_path, "cameras.npz")
    cam_dict = np.load(cam_file)
    out = cv2.decomposeProjectionMatrix(cam_dict["world_mats"][i])
    K = torch.tensor(out[0]).view(1,3,3).float()
    R = torch.tensor(out[1]).view(1,3,3).float()
    t = torch.tensor(out[2])
    t = (t[:3] / t[3])[:, 0].view(1,3).float()
    image_size = [480, 640]

    # For qualitative testing, uncomment the lines below
    depth_map, normal_map, vertex_map = render_depth(R,t,K,verts, faces, image_size, device)
    plt.figure()
    plt.imshow(depth_map.detach().cpu()) # (h,w)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth image')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(f'{vis_dir}/{str(i).zfill(6)}.jpg')