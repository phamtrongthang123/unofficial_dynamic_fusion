import os
import argparse
import numpy as np
import torch
import cv2
import trimesh
import warnings
warnings.simplefilter("ignore", UserWarning)
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
from tmp_utils import SE3, decompose_se3, blending, get_diag, dqnorm, custom_transpose_batch, get_W, render_depth, average_edge_dist_in_face, robust_Tukey_penalty, warp_helper, warp_to_live_frame, plot_vis_depthmap, SE3_dq
import time 
from functorch import vmap, jacrev
from icp import compute_normal, compute_vertex



def data_term(gt_Xc, Tlw, dgv, dgse, dgw, node_to_nn):
    Xc = gt_Xc[2]
    nc = gt_Xc[3]
    gt_v = gt_Xc[0]
    gt_n = gt_Xc[1]
    Xt = warp_helper(Xc, Tlw, dgv, dgse, dgw, node_to_nn)
    # if you want it to similar to surfel warp, uncomment the lines below
    e = gt_n.dot(Xt - gt_v).view(1,1)**2
    re = e
    # or using tukey like in DynFu paper.
    # e = gt_n.dot(Xt - gt_v).view(1,1)
    # re = robust_Tukey_penalty(e, 0.01)
    return re


def reg_term(Tlw, dgv, dgse, dgw, dgv_nn):
    shape = dgv.shape
    warp_helper_vmap = vmap(warp_helper, in_dims=(0, None, None, None, None, 0))
    dgv_warped = warp_helper_vmap(dgv, Tlw, dgv, dgse, dgw, dgv_nn)
    dgv_nn_t = dgv_warped[dgv_nn]
    assert dgv_nn_t.shape == (shape[0], 4,3)
    dgv_warped_t = dgv_warped.view(-1, 1, 3).repeat_interleave(4,1)
    el2 = torch.linalg.norm(dgv_warped_t - dgv_nn_t, dim = 2)**2  
    assert el2.shape == (shape[0], 4) 
    ssel2 = el2.mean()
    return ssel2  



def energy(gt_Xc, Tlw, dgv, dgse, dgw, node_to_nn, dgv_nn):
    data_vmap = vmap(data_term, in_dims=(0, None, None, None, None, 0))
    data_val = data_vmap(gt_Xc, Tlw, dgv, dgse, dgw, node_to_nn).mean()
    reg_val = reg_term(Tlw, dgv, dgse, dgw, dgv_nn)
    # re = ((data_val + 5*reg_val) / 2).float() # 5 = lambda in surfelwarp paper
    re = data_val
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
                dists, idx = kdtree.query(vertex_map[y,x].cpu(), k=knn, workers=-1)
                node_to_nn.append(torch.tensor(idx).type_as(vertex_map).long())
                res.append(torch.stack([vertex0[y,x], normal0[y,x], vertex_map[y,x], normal_map[y,x]]))
    node_to_nn = torch.stack(node_to_nn)
    dists, idx = kdtree.query(dgv.cpu(), k=range(2,2+knn), workers=-1)
    dgv_nn = torch.tensor(idx).type_as(dgv).long()
    res = torch.stack(res)
    assert len(res.shape) == 3 # filtered_dim, 4, 3 
    assert len(node_to_nn.shape) == 2 # filtered_dim, 4 
    energy_jac = jacrev(energy, argnums=3, has_aux=True)
    dqnorm_vmap = vmap(dqnorm, in_dims=0)
    print("Start optimize! ")
    I = torch.eye(8).type_as(Tlw) # to counter not full rank
    bs = 1 # res.shape[0]
    lmda = 5
    for i in range(5):
        jse3,fx = energy_jac(res, Tlw, dgv, dgse, dgw, node_to_nn, dgv_nn)
        # print("done se3")
        j = jse3.view(bs, len(dgv),1,8) # [bs,n_node,1,8]
        jT = custom_transpose_batch(j,isknn=True) # [bs,n_node, 8,1]
        tmp_A = torch.einsum('bnij,bnjl->bnil',jT, j).view(bs*len(dgv),8,8) # [bs*n_node,8,8]    
        A = (tmp_A + lmda*I.view(1,1,8,8)).view(bs*len(dgv),8,8)  #  [bs*n_node,8,8]
        b = torch.einsum('bnij,bj->bni', jT,fx.view(bs,1)).view(bs*len(dgv),8,1) # [bs*n_node, 8, 1]
        solved_delta = torch.linalg.lstsq(A, b)
        solved_delta = solved_delta.solution.view(bs,len(dgv),8).mean(dim=0) 
        # eliminate nan and inf 
        solved_delta = torch.where(torch.any(torch.isnan(solved_delta.view(dgse.shape))).view(-1,1), torch.zeros(8).type_as(dgse), solved_delta.view(dgse.shape))
        solved_delta = torch.where(torch.any(torch.isinf(solved_delta.view(dgse.shape))).view(-1,1), torch.zeros(8).type_as(dgse), solved_delta.view(dgse.shape))
        # update 
        dgse -=  solved_delta
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
                partial_tsdf.export(os.path.join(args.save_dir, f"mesh_{str(i).zfill(6)}.obj"))
                # warp vertices to live frame 
                verts = warp_to_live_frame(verts, Tlw_i, self.dgv, self.dgse, self.dgw,  self._kdtree)
                partial_tsdf = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
                partial_tsdf.export(os.path.join(args.save_dir, f"warped_mesh_{str(i).zfill(6)}.obj"))
                # render depth, vertex and normal 
                image_size = [H,W]
                # we already use Tlw_i in warping, so no need of passing it into rendering function
                pose_tmp = torch.eye(4).to(self.device)
                depth_map, normal_map, vertex_map = render_depth(pose_tmp[:3,:3].view(1,3,3),pose_tmp[:3,3].view(1,3),K.view(1,3,3),verts, faces, image_size, self.device) 

                plot_vis_depthmap(depth_map, f'{args.save_dir}/vis', i)

                T10 = self.icp_tracker(depth0, depth_map, K)  # transform from 0 to 1
                Tlw = Tlw @ T10
                Tlw_i = torch.inverse(Tlw).to(self.device)
                # optim energy and set dgse 
                self.dgse = optim_energy(depth0, depth_map, normal_map, vertex_map,Tlw_i, self.dgv, self.dgse, self.dgw, self._kdtree, K)

                # update Tlw 

            # fusion
            if i ==0:
                self.tsdf_volume.integrate(depth0,
                                    K,
                                    Tlw,
                                    obs_weight=1.,
                                    color_img=color0
                                    )
            # else:
            #     self.tsdf_volume.integrate_dynamic(depth0, 
            #                         self.dgv,
            #                         self.dgse,
            #                         self.dgw,
            #                         self._kdtree,
            #                         K,
            #                         Tlw,
            #                         obs_weight=1.,
            #                         color_img=color0)
            t1 = get_time()
            t += [t1 - t0]
            print("processed frame: {:d}, time taken: {:f}s".format(i, t1 - t0))
            poses += [Tlw.cpu().numpy()]
            poses_gt += [pose_gt.cpu().numpy()]
            if i == 0:
                self.construct_graph()
                print(f"Done, now we have {self.dgv.shape[0]} nodes!")
            else:
                print("Start update graph!")
                try:
                    # self.update_graph(Tlw)
                    print(f"Done, now we have {self.dgv.shape[0]} nodes!")
                except Exception as e:
                    print(f'Failed to update graph because of {e}! ') 
                # pass
            if i == 420: break
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
        partial_tsdf.export(os.path.join(args.save_dir, "mesh.obj"))
        np.savez(os.path.join(args.save_dir, "traj.npz"), poses=self.poses)
        np.savez(os.path.join(args.save_dir, "traj_gt.npz"), poses=self.poses_gt)


    def construct_graph(self):
        verts, faces, norms = self.tsdf_volume.get_mesh()
        # average_distances = []
        # for f in faces:
        #     average_distances.append(average_edge_dist_in_face(f, verts))
        # self._radius = self.subsample_rate * np.average(np.array(average_distances))
        self._radius = 0.01 # the paper said e = 0.01 in experiment
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
        self._kdtree = KDTree(nodes_v.astype(np.float32))
    def update_graph(self, Tlw):
        verts_c, faces, norms = self.tsdf_volume.get_mesh()
        dists, idx = self._kdtree.query(verts_c, k=4, workers=-1)
        verts_c = torch.tensor(verts_c).to(self.device)
        verts_c_nn_idx = torch.tensor(idx).long()
        verts_c_nn = self.dgv[verts_c_nn_idx]
        verts_c_nn_dgw = self.dgw[verts_c_nn_idx]
        verts_c_rep = verts_c.view(-1,1,3).repeat_interleave(4,dim=1)
        verts_c_l2_dgw = torch.linalg.norm(verts_c_nn - verts_c_rep, dim=2) / verts_c_nn_dgw
        mask_unsupported = torch.min(verts_c_l2_dgw, dim = 1).values >= 1
        def get_se3_helper(Xc, Tlw, dgv, dgse, dgw, node_to_nn):
            dgv_nn = dgv[node_to_nn]
            dgw_nn = dgw[node_to_nn]
            dgse_nn = dgse[node_to_nn]
            assert dgse_nn.shape == (4,8)
            assert dgv_nn.shape == (4,3)
            T = get_W(Xc, Tlw, dgse_nn, dgw_nn, dgv_nn)
            print(T)
            re_dq = dqnorm(SE3_dq(T.view(4,4)))
            print(re_dq)
            return re_dq.view(8)
        get_se3_helper_vmap = vmap(get_se3_helper, in_dims=(0, None, None, None, None, 0))
        nodes_v, nodes_idx = uniform_sample(verts_c[mask_unsupported].cpu(), self._radius)
        dists, idx = self._kdtree.query(nodes_v, k=4, workers=-1)
        nodes_v = torch.tensor(nodes_v).to(self.device)
        nodes_v_nn_idx = torch.tensor(idx).long().to(self.device)
        nodes_se3 = get_se3_helper_vmap(nodes_v, Tlw, self.dgv, self.dgse, self.dgw, nodes_v_nn_idx).float().to(self.device)
        assert nodes_se3.shape[0] == nodes_v.shape[0]
        nodes_w = torch.tensor(2.0*self._radius).view(1,1).repeat_interleave(nodes_v.shape[0], 0).to(self.device).view(-1)

        self.dgv = torch.cat((self.dgv,nodes_v), dim = 0).to(self.device)
        self.dgse = torch.cat((self.dgse,nodes_se3), dim = 0).to(self.device)
        self.dgw = torch.cat((self.dgw,nodes_w), dim = 0).float().to(self.device)
        assert self.dgv.shape[0] == self.dgw.shape[0] and self.dgv.shape[0] == self.dgse.shape[0], f'{self.dgv.shape, self.dgw.shape, self.dgse.shape, nodes_w.shape, nodes_v.shape}'

        # construct kd tree
        self._kdtree = KDTree(self.dgv.cpu())
        # pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # standard configs
    parser.add_argument('--config', type=str, default="configs/fr1_desk.yaml", help='Path to config file.')
    parser.add_argument("--save_dir", type=str, default=None, help="Directory of saving results.")
    args = load_config(parser.parse_args())
    os.makedirs(args.save_dir, exist_ok=True)
    dynfu = DynFu(args)
    dynfu.process()    
    if args.save_dir is not None:
        dynfu.save_mesh()

