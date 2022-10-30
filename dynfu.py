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
from tmp_utils import SE3, decompose_se3, blending, get_diag, dqnorm, custom_transpose_batch, get_W, render_depth, average_edge_dist_in_face, robust_Tukey_penalty
import time 
from functorch import vmap, jacrev
from icp import compute_normal, compute_vertex
def warp_helper(Xc, Tlw, dgv, dgse, dgw, node_to_nn):
    dgv_nn = dgv[node_to_nn]
    dgw_nn = dgw[node_to_nn]
    dgse_nn = dgse[node_to_nn]
    assert dgse_nn.shape == (4,8)
    assert dgv_nn.shape == (4,3)
    T = get_W(Xc, Tlw, dgse_nn, dgw_nn, dgv_nn)
    R, t= decompose_se3(T.type_as(Xc))
    Xt =  (torch.einsum('bij,bjk->bi', R, Xc.view(1,3,1)) + t.squeeze(-1)).squeeze() # shape [3] == target
    
    return Xt 
def warp_to_live_frame(verts, Tlw, dgv, dgse, dgw,  kdtree): 
    assert len(verts.shape) == 2 and verts.shape[1] == 3 
    knn = 4
    node_to_nn = []
    for i in range(verts.shape[0]):
        dists, idx = kdtree.query(verts[i], k=knn)
        node_to_nn.append(torch.tensor(idx))
    node_to_nn = torch.stack(node_to_nn)
    vmap_helper = vmap(warp_helper, in_dims=(0, None,None,None,None, 0))
    verts_live = vmap_helper(verts, Tlw, dgv, dgse, dgw, node_to_nn)
    assert len(verts_live.shape) == 2 and verts_live.shape[1] == 3
    return verts_live

def data_term(gt_Xc, Tlw, dgv, dgse, dgw, node_to_nn):
    Xc = gt_Xc[2]
    nc = gt_Xc[3]
    gt_v = gt_Xc[0]
    gt_n = gt_Xc[1]
    Xt = warp_helper(Xc, Tlw, dgv, dgse, dgw, node_to_nn)
    e = gt_n.dot(Xt - gt_v).view(1,1)
    # re = e
    re = robust_Tukey_penalty(e, 0.01)
    return re, re


def optim_energy(depth0, depth_map, normal_map, vertex_map,Tlw, dgv, dgse, dgw, kdtree, K):
    knn = 4
    vertex0 = compute_vertex(depth0, K)
    normal0 = compute_normal(vertex0)
    mask0 = depth0 > 0. 
    mask_hat = depth_map > 0. 
    # for data term 
    assert mask_hat.shape == (480, 640)
    H, W = mask0.shape 
    res = []
    node_to_nn = []
    for y in range(H): 
        for x in range(W):
            if mask0[y,x] and mask_hat[y,x]:
                dists, idx = kdtree.query(vertex_map[y,x].cpu(), k=knn)
                node_to_nn.append(torch.tensor(idx).type_as(vertex_map).long())
                res.append(torch.stack([vertex0[y,x], normal0[y,x], vertex_map[y,x], normal_map[y,x]]))
    node_to_nn = torch.stack(node_to_nn)
    res = torch.stack(res)
    assert len(res.shape) == 3 # filtered_dim, 4, 3 
    assert len(node_to_nn.shape) == 2 # filtered_dim, 4 
    bs = res.shape[0]
    vmap_data_jac = vmap(jacrev(data_term, argnums=3, has_aux=True), in_dims=(0, None, None, None, None, 0))
    dqnorm_vmap = vmap(dqnorm, in_dims=0)
    print("Start optimize! ")
    I = torch.eye(8).type_as(Tlw) # to counter not full rank
    lmda = 1e-4
    for i in range(5):
        jse3,fx = vmap_data_jac(res, Tlw, dgv, dgse, dgw, node_to_nn)
        # print("done se3")
        j = jse3.view(bs, len(dgv),1,8) # [bs,n_node,1,8]
        jT = custom_transpose_batch(j,isknn=True) # [bs,n_node, 8,1]
        tmp_A = torch.einsum('bnij,bnjl->bnil',jT, j) # [bs,n_node,8,8]
        A = (tmp_A + lmda*I.view(1,1,8,8)).view(bs*len(dgv),8,8)  #  [bs*n_node,8,8]
        b = torch.einsum('bnij,bj->bni', jT,fx.view(bs,1)).view(bs*len(dgv),8,1) # [bs*n_node, 8, 1]
        solved_delta = torch.linalg.lstsq(A, b)
        solved_delta = solved_delta.solution.view(bs,len(dgv),8).mean(dim=0) 
        dgse -=  solved_delta.view(dgse.shape) # set 0.2 here helps
        dgse = dqnorm_vmap(dgse.view(-1,8)).view(len(dgv),8)
        print("log: ", torch.sum(fx), torch.sum(fx.abs()), torch.mean(fx), torch.mean(fx.abs()))
    return dgse

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
        Tlw, depth1, color1 = None, None, None
        for i in range(0, len(self.dataset), 1):
            t0 = get_time()
            sample = self.dataset[i]
            color0, depth0, pose_gt, K = sample  # use live image as template image (0)
            # depth0[depth0 <= 0.5] = 0.

            if i == 0:  # initialize
                Tlw = pose_gt
            else:  # tracking

                Tlw_i = torch.inverse(Tlw).to(self.device)
                verts, faces, norms = self.tsdf_volume.get_mesh(istorch=True)
                partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
                partial_tsdf.export(os.path.join(args.save_dir, f"mesh{i}.ply"))
                # warp vertices to live frame 
                verts = warp_to_live_frame(verts, Tlw_i, self.dgv, self.dgse, self.dgw,  self._kdtree)
                partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
                partial_tsdf.export(os.path.join(args.save_dir, f"mesh{i}_warped.ply"))
                # render depth, vertex and normal 
                image_size = [H,W]
                depth_map, normal_map, vertex_map = render_depth(Tlw_i[:3,:3].view(1,3,3),Tlw_i[:3,3].view(1,3),K.view(1,3,3),verts, faces, image_size, self.device)      
                T10 = self.icp_tracker(depth0, depth_map, K)  # transform from 0 to 1
                Tlw = Tlw @ T10
                Tlw_i = torch.inverse(Tlw).to(self.device)
                # optim energy and set dgse 
                self.dgse = optim_energy(depth0, depth_map, normal_map, vertex_map,Tlw_i, self.dgv, self.dgse, self.dgw, self._kdtree, K)

                # update Tlw 

            # fusion
            self.tsdf_volume.integrate(depth0,
                                K,
                                Tlw,
                                obs_weight=1.,
                                color_img=color0
                                )
            
            t1 = get_time()
            t += [t1 - t0]
            print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
            poses += [Tlw.cpu().numpy()]
            poses_gt += [pose_gt.cpu().numpy()]
            if i == 0:
                self.construct_graph()
            else:
                # self.update_graph() function 
                pass
            if i == 4: break
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
        color0, depth0, pose_gt, K = self.dataset[0]
        verts, faces, norms = self.tsdf_volume.get_mesh(istorch=True)
        # warp vertices to live frame 
        Tlw = torch.tensor(self.poses[-1])
        verts = warp_to_live_frame(verts, Tlw, self.dgv.cpu(), self.dgse.cpu(), self.dgw.cpu(),  self._kdtree)
        # render depth, vertex and normal 
        H, W = self.dataset.H, self.dataset.W
        image_size = [H,W]
        Tlw = Tlw.to(self.device)
        depth_map, normal_map, vertex_map = render_depth(Tlw[:3,:3].view(1,3,3),Tlw[:3,3].view(1,3),K.view(1,3,3),verts, faces, image_size, self.device)      
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(depth_map.detach().cpu()) # (h,w)
        plt.colorbar(label='Distance to Camera')
        plt.title('Depth image')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        plt.show()

    def construct_graph(self):
        verts, faces, norms = self.tsdf_volume.get_mesh()
        average_distances = []
        for f in faces:
            average_distances.append(average_edge_dist_in_face(f, verts))
        self._radius = self.subsample_rate * np.average(np.array(average_distances))
        # print(verts.shape, faces.shape, self._radius)
        self._vertices = verts 
        self._faces = faces
        nodes_v, nodes_idx = uniform_sample(self._vertices, self._radius)
        self.dgv = torch.tensor(nodes_v.copy()).to(self.device)
        # define list of node warp
        dgse = []  
        dgw = []
        for j in range(len(self.dgv)):
            dgse.append(torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]).float())
            dgw.append(torch.tensor(2.0*self._radius))
        self.dgse = torch.stack(dgse).to(self.device)
        self.dgw = torch.stack(dgw).float().to(self.device)

        # construct kd tree
        self._kdtree = KDTree(nodes_v)
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

