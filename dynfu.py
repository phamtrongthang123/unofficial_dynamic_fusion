import os
import argparse
import numpy as np
import torch
import cv2
import trimesh
from matplotlib import pyplot as plt
from fusion import TSDFVolumeTorch
from dataset.ourdataset import OurDataset
from tracker import ICPTracker
from utils import load_config, get_volume_setting, get_time
from numpy import linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from tmp_utils import cal_dist, uniform_sample
from numba import njit
import time 
class DynFu():
    def __init__(self, args) -> None:
        self.subsample_rate = 5.0
        self.args = args 
        self.knn = 4 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        self.dataset = OurDataset(os.path.join(args.data_root), self.device, near=args.near, far=args.far, img_scale=1.)
        

        self.vol_dims, self.vol_origin, self.voxel_size = get_volume_setting(args)
        self.tsdf_volume = TSDFVolumeTorch(self.vol_dims, self.vol_origin, self.voxel_size, self.device, margin=3, fuse_color=args.fuse_color)
        self.icp_tracker = ICPTracker(args, self.device)
    def process(self):
        H, W = self.dataset.H, self.dataset.W
        t, poses, poses_gt = list(), list(), list()
        curr_pose, depth1, color1 = None, None, None
        for i in range(0, len(self.dataset), 1):
            t0 = get_time()
            sample = self.dataset[i]
            color0, depth0, pose_gt, K = sample  # use live image as template image (0)
            # depth0[depth0 <= 0.5] = 0.

            if i == 0:  # initialize
                curr_pose = pose_gt
            else:  # tracking
                # 1. render depth image (1) from tsdf volume
                depth1, color1, vertex01, normal1, mask1 = self.tsdf_volume.render_model(curr_pose, K, H, W, near=args.near, far=args.far, n_samples=args.n_steps)
                T10 = self.icp_tracker(depth0, depth1, K)  # transform from 0 to 1
                curr_pose = curr_pose @ T10
                

            # fusion
            self.tsdf_volume.integrate(depth0,
                                K,
                                curr_pose,
                                obs_weight=1.,
                                color_img=color0
                                )
            
            t1 = get_time()
            t += [t1 - t0]
            print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
            poses += [curr_pose.cpu().numpy()]
            poses_gt += [pose_gt.cpu().numpy()]
            if i == 0:
                self.construct_graph()
            else:
                # self.update_graph() function 
                pass
            if i > 1: break
        avg_time = np.array(t).mean()
        print("average processing time: {:f}s per frame, i.e. {:f} fps".format(avg_time, 1. / avg_time))
        # compute tracking ATE
        self.poses_gt = np.stack(poses_gt, 0)
        self.poses = np.stack(poses, 0)
        traj_gt = np.array(poses_gt)[:, :3, 3]
        traj = np.array(poses)[:, :3, 3]
        rmse = np.sqrt(np.mean(np.linalg.norm(traj_gt - traj, axis=-1) ** 2))
        print("RMSE: {:f}".format(rmse))

    def save_mesh(self):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        if args.fuse_color:
            verts, faces, norms, colors = self.tsdf_volume.get_mesh()
            partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms, vertex_colors=colors)
        else:
            verts, faces, norms = self.tsdf_volume.get_mesh()
            partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
        partial_tsdf.export(os.path.join(args.save_dir, "mesh.ply"))
        np.savez(os.path.join(args.save_dir, "traj.npz"), poses=self.poses)
        np.savez(os.path.join(args.save_dir, "traj_gt.npz"), poses=self.poses_gt)

    def average_edge_dist_in_face(self, f, verts):
        v1 = verts[f[0]]
        v2 = verts[f[1]]
        v3 = verts[f[2]]
        return (cal_dist(v1,v2) + cal_dist(v1,v3) + cal_dist(v2,v3))/3
    def construct_graph(self):
        verts, faces, norms = self.tsdf_volume.get_mesh()
        average_distances = []
        for f in faces:
            average_distances.append(self.average_edge_dist_in_face(f, verts))
        self._radius = self.subsample_rate * np.average(np.array(average_distances))
        # print(verts.shape, faces.shape, self._radius)
        self._vertices = verts 
        self._faces = faces
        nodes_v, nodes_idx = uniform_sample(self._vertices, self._radius)
        # define list of node warp
        self._nodes = []  
        for j in range(len(nodes_v)):
            self._nodes.append((nodes_idx[j],
                                nodes_v[j],
                                np.array([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00], dtype=np.float32),
                                2 * self._radius))

        # construct kd tree
        self._kdtree = KDTree(nodes_v)
        # cache all neighbor node for all vertices in canonical space
        self._neighbor_look_up = []
        for vert in verts:
            dists, idx = self._kdtree.query(vert, k=self.knn)
            self._neighbor_look_up.append(idx) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/fr1_desk.yaml", help='Path to config file.')
    parser.add_argument("--save_dir", type=str, default=None, help="Directory of saving results.")
    args = load_config(parser.parse_args())
    dynfu = DynFu(args)
    dynfu.process()
    
    if args.save_dir is not None:
        dynfu.save_mesh()

