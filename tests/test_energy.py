import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch
from scipy.spatial import KDTree
from numpy import linalg as la
import time 
from tmp_utils import SE3, decompose_se3, blending, get_diag, dqnorm, custom_transpose_batch, get_W, render_depth
import numpy as np 
from functorch import vmap, vjp, jacrev, jacfwd, hessian, jvp, grad
from einops import rearrange, reduce, repeat


def test_energy_small():
    _ = torch.manual_seed(0)
    _radius = torch.tensor(1)
    dgv = torch.randn(52,3)
    dgv = torch.tensor([[0,0.2,0],[0.1,0.2,0],[0.3,0.2,0], [0.4,0.2,0],[0.5,0.2,1], [0.6,0.2,0.1], [0.7,0.2,0.1], [0.8,0.2,0.1], [0.9,0.2,0.1]])*10
    dgse = []  
    dgw = []
    for j in range(len(dgv)):
        dgse.append(torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01]).float())
        dgw.append(torch.tensor(2.0*_radius))
    dgse = torch.stack(dgse)
    dgw = torch.stack(dgw)
    _kdtree = KDTree(dgv)
    bs = 2 
    knn = 4
    Xc = torch.randn(bs,3).float()*3
    Xc = torch.tensor([[0.2,0,0],[0.7,0,0.1]]).float()*10
    target = torch.ones(bs,3).float()
    target = torch.tensor([[0.2,0.2,0], [0.65,0.2,0.1]]).float()*10
    Tlw = torch.eye(4)
    node_to_nn = []
    for i in range(Xc.shape[0]):
        dists, idx = _kdtree.query(Xc[i], k=knn)
        node_to_nn.append(torch.tensor(idx))
    node_to_nn = torch.stack(node_to_nn)
    assert node_to_nn.shape == (bs,4)
    def energy(Xc, target, Tlw, dgv, dgse, dgw, node_to_nn):
        dgv_nn = dgv[node_to_nn]
        dgw_nn = dgw[node_to_nn]
        dgse_nn = dgse[node_to_nn]
        assert dgse_nn.shape == (4,8)
        assert dgv_nn.shape == (4,3)
        T = get_W(Xc, Tlw, dgse_nn, dgw_nn, dgv_nn)
        R, t= decompose_se3(T)
        Xt =  (torch.einsum('bij,bjk->bi', R, Xc.view(1,3,1)) + t.squeeze(-1)).squeeze() # shape [3] == target 
        return torch.linalg.norm(Xt-target), torch.linalg.norm(Xt-target)
    def solve_(A,b):
        solved_delta = torch.linalg.lstsq(A.view(8,8), b.view(8,1))
        res = solved_delta.solution 
        return res
    # use vmap to avoid linear combination in batch dimension 
    compute_batch_jacobian = vmap(jacrev(energy, argnums=4, has_aux=True), in_dims=(0, 0, None,None,None,None, 0))
    solve_vmap = vmap(solve_,in_dims=(0,0))
    dqnorm_vmap = vmap(dqnorm, in_dims=0)
    lmda = 1e-6
    for i in range(5):
        jse3,fx = compute_batch_jacobian(Xc, target, Tlw, dgv, dgse, dgw, node_to_nn) 
        j = jse3.view(bs, len(dgv),1,8) # [bs,n_node,1,8], each result is scalar so the shape is the same as dgse. Note that we need to keep 1 there because pytorch squeezed out our scalar from fx.
        jT = custom_transpose_batch(j,isknn=True) # [bs,n_node, 8,1]
        tmp_A = torch.einsum('bnij,bnjl->bnil',jT, j) # [bs,n_node,8,8]
        A = (tmp_A + lmda*get_diag(tmp_A)).view(bs*len(dgv),8,8)  #  [bs*n_node,8,8]
        # error is a number for each batch, so we only need to multiply it inside normally 
        b = torch.einsum('bnij,bj->bni', jT,fx.view(bs,1)).view(bs*len(dgv),8,1) # [bs*n_node, 8, 1]
        solved_delta = solve_vmap(A, b).view(bs,len(dgv),8).sum(dim=0) # [bs*n_node,8,1]
        dgse -=  0.2*solved_delta.view(dgse.shape) # set 0.2 here helps
        dgse = dqnorm_vmap(dgse.view(-1,8)).view(len(dgv),8)
        print("log: ", torch.sum(fx), torch.sum(solved_delta.abs()), torch.linalg.norm(A)) # if this decreases each time, then we are success, probably .-. 
    
    for jj in range(bs):
        dgv_nn = dgv[node_to_nn[jj]]
        dgw_nn = dgw[node_to_nn[jj]]
        dgse_nn = dgse[node_to_nn[jj]]
        assert dgse_nn.shape == (4,8)
        assert dgv_nn.shape == (4,3)
        T = get_W(Xc[jj], Tlw, dgse_nn, dgw_nn, dgv_nn)
        R, t= decompose_se3(T)
        Xt =  (torch.einsum('bij,j->bi', R, Xc[jj]) + t.squeeze(-1)).squeeze()
        print(Xt)
        print(R)
        print(t)
        assert torch.allclose(Xt, target[jj], rtol=1e-1, atol=1e-02), f"{Xt} is different from {target[jj]}" # should pass

