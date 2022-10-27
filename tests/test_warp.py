import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from skimage import measure
from skimage.draw import ellipsoid
import trimesh 
import cv2 
import torch
from scipy.spatial import KDTree
from numpy import linalg as la
import time 
from tmp_utils import SE3, decompose_se3, cal_dist, uniform_sample, average_edge_dist_in_face


def blending(Xc, dqs, dgw, dgv):
    repxc = Xc.view(Xc.shape[0], 1, Xc.shape[1]).repeat_interleave(dgv.shape[1],1)
    assert repxc.shape == dgv.shape 
    wixc = torch.exp(torch.linalg.norm(dgv - repxc, dim=2) / (2*torch.pow(dgw,2)))
    assert wixc.shape == dgw.shape
    qkc = torch.einsum('bi,bik->bk',wixc,dqs)
    assert qkc.shape == (dqs.shape[0],8)
    dem = torch.linalg.norm(qkc,dim=1).view(dqs.shape[0], 1)
    return qkc/dem 

def get_W(Xc, Tlw, dqs, dgw, dgv):
    dqb = blending(Xc, dqs, dgw, dgv).view(Xc.shape[0],8)
    T = torch.einsum('ij,bjk -> bik',Tlw, SE3(dqb))
    return T 

def test_get_W():
    _radius = torch.tensor(0.1)
    nodes_v_t = torch.randn(52,3)
    _nodes = []  
    for j in range(len(nodes_v_t)):
        _nodes.append([nodes_v_t[j],
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # real Tic = T_k-1
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # tmp T_k for optim
                            2 * _radius])
    _kdtree = KDTree(nodes_v_t)
    bs = 5 
    knn = 4
    Xc = torch.rand(bs,3).float()
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
    # xc [bs, 3], neighbor [bs, 8], [bs,3] and [bs, 1]. Chi argnum = bs,8 de lay jac. sau khi tinh xong thi concate ket qua vo jac cuoi cung hoac GN luon neu muon 
    T = get_W(Xc, Tlw, dqs, dgw, dgv)
    R, t= decompose_se3(T)
    Xt =  (torch.einsum('bij,bj->bi', R, Xc) + t.squeeze(-1))
    
    assert Xt.shape == (bs,3)