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



def test_get_W():
    _radius = torch.tensor(1)
    nodes_v_t = torch.randn(52,3)
    _nodes = []  
    for j in range(len(nodes_v_t)):
        _nodes.append([nodes_v_t[j],
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # real Tic = T_k-1
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # tmp T_k for optim
                            2 * _radius])
    _kdtree = KDTree(nodes_v_t)
    bs = 1 
    knn = 2
    Xc = torch.randn(bs,3).float()
    target = torch.ones(bs,3).float()
    Tlw = torch.eye(4)
    dqs = []
    dgw = [] 
    dgv = []
    for i in range(Xc.shape[0]):
        dists, idx = _kdtree.query(Xc[i], k=knn)
        dq = [] 
        dw = [] 
        dv = []
        for id in idx:
            dv.append(_nodes[i][0])
            dq.append(_nodes[i][1])
            dw.append(_nodes[i][3])
        dqs.append(torch.stack(dq))
        dgw.append(torch.stack(dw))
        dgv.append(torch.stack(dv))
    dqs = torch.stack(dqs)
    dgw = torch.stack(dgw)
    dgv = torch.stack(dgv)
    assert dqs.shape == (bs,knn,8)
    assert dgw.shape == (bs,knn)
    assert dgv.shape == (bs,knn,3)
    # theo toan thi moi W là 1 xc + bo neighbors, nen neu batch xc thi + batch bo neighbor. Vectorize duoc. 
    # xc [bs, 3], neighbor [bs, 8], [bs,3] and [bs, 1]. 
    # we will use vmap so let's assume we pass each input inside
    T = get_W(Xc[0], Tlw, dqs[0], dgw[0], dgv[0])
    R, t= decompose_se3(T)
    Xt =  (torch.einsum('bij,bj->bi', R, Xc) + t.squeeze(-1))
    
    assert Xt.shape == (bs,3)

def test_dqnorm():
    a = torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.0]).float()
    a_norm = dqnorm(a)
    assert torch.allclose(a, a_norm)

def test_GN_blendingW():
    _ = torch.manual_seed(0)

    # note that when running in real code, this _radius control the dgw inside w, this could cause the least square fail solve equation. I set 0.1 fail, so i set it = 1 here.
    # the exp is sensitive, if the element is too large, the dual blending will break. 
    _radius = torch.tensor(1)
    nodes_v_t = torch.randn(52,3)
    nodes_v_t = torch.tensor([[0,0.2,0], [0.4,0.2,0],[0.5,0.2,1], [0.8,0.2,0.1]])
    _nodes = []  
    for j in range(len(nodes_v_t)):
        _nodes.append([nodes_v_t[j],
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01]).float(), # real Tic = T_k-1
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.0]).float(), # tmp T_k for optim
                            2 * _radius])
    _kdtree = KDTree(nodes_v_t)
    bs = 2 
    knn = 2
    Xc = torch.randn(bs,3).float()*3
    Xc = torch.tensor([[0.2,0,0],[0.7,0,0.1]]).float()
    target = torch.ones(bs,3).float()
    target = torch.tensor([[0.2,0.2,0], [0.65,0.2,0.1]]).float()
    Tlw = torch.eye(4)
    dqs = []
    dgw = [] 
    dgv = []
    for i in range(Xc.shape[0]):
        dists, idx = _kdtree.query(Xc[i], k=knn)
        dq = [] 
        dw = [] 
        dv = []
        print(idx)
        for id in idx:
            dv.append(_nodes[i][0])
            dq.append(_nodes[i][1])
            dw.append(_nodes[i][3])
        dqs.append(torch.stack(dq))
        dgw.append(torch.stack(dw))
        dgv.append(torch.stack(dv))
    dqs = torch.stack(dqs)
    dgw = torch.stack(dgw)
    dgv = torch.stack(dgv)
    assert dqs.shape == (bs,knn,8)
    assert dgw.shape == (bs,knn)
    assert dgv.shape == (bs,knn,3)
    # theo toan thi moi W là 1 xc + bo neighbors, nen neu batch xc thi + batch bo neighbor. Vectorize duoc. 
    # xc [bs, 3], neighbor [bs, 8], [bs,3] and [bs, 1]. Chi argnum = bs,8 de lay jac. sau khi tinh xong thi concate ket qua vo jac cuoi cung hoac GN luon neu muon 
    def energy(Xc, Tlw, dqs, dgw, dgv, target):
        T = get_W(Xc, Tlw, dqs, dgw, dgv)
        R, t= decompose_se3(T)
        Xt =  (torch.einsum('bij,bjk->bi', R, Xc.view(1,3,1)) + t.squeeze(-1)).squeeze() # shape [3] == target 
        return torch.linalg.norm(Xt-target), torch.linalg.norm(Xt-target)
    def solve_(A,b):
        solved_delta = torch.linalg.lstsq(A.view(8,8), b.view(8,1))
        res = solved_delta.solution 
        return res
    # use vmap to avoid linear combination in batch dimension 
    compute_batch_jacobian = vmap(jacrev(energy, argnums=2, has_aux=True), in_dims=(0, None, 0,0,0,0))
    solve_vmap = vmap(solve_,in_dims=(0,0))
    dqnorm_vmap = vmap(dqnorm, in_dims=0)
    lmda = 1e-6
    for i in range(5):
        jse3,fx = compute_batch_jacobian(Xc,Tlw, dqs, dgw, dgv, target)   
        j = jse3.view(bs, knn, 1,8) # [bs,knn,1,8], because we return norm / scalar so the jac has the same shape as input dqs
        jT = custom_transpose_batch(j,isknn=True) # [bs,knn,8,1]
        tmp_A = torch.einsum('bkij,bkjl->bkil',jT, j)
        A = (tmp_A + lmda*get_diag(tmp_A)).view(bs*knn,8,8)  # + res term # [bs,knn,8,8]
        # error is a number for each batch, so we only need to multiply it inside normally 
        b = torch.einsum('bkij,bj->bki', jT,fx.view(bs,1)).view(bs*knn,8,1) # [bs, knn,8, 1]
        solved_delta = solve_vmap(A, b) # [bs*knn,8,1]
        dqs -=  0.2*solved_delta.view(dqs.shape) # set 0.2 here helps
        dqs = dqnorm_vmap(dqs.view(-1,8)).view(bs,knn,8)
        print("log: ", torch.sum(fx), torch.sum(solved_delta.abs()), torch.linalg.norm(A)) # if this decreases each time, then we are success, probably .-. 
    
    for jj in range(bs):
        T = get_W(Xc[jj], Tlw, dqs[jj], dgw[jj], dgv[jj])
        R, t= decompose_se3(T)
        Xt =  (torch.einsum('bij,j->bi', R, Xc[jj]) + t.squeeze(-1)).squeeze()
        print(Xt)
        print(R)
        print(t)
        assert torch.allclose(Xt, target[jj], rtol=1e-1, atol=1e-02), f"{Xt} is different from {target[jj]}" # should pass

