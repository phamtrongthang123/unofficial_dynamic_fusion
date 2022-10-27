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


def blending(xc, nodes, kdtree):
    assert xc.shape == (1,3)
    xc = xc.view(3)
    dists, idx = kdtree.query(xc, k=4)
    numerator = torch.zeros(8)
    for i in idx: 
        wixc = torch.exp(torch.linalg.norm(nodes[i][1] - xc) / (2*nodes[i][-1]**2))
        qkc = nodes[i][2]*wixc 
        numerator += qkc 
    dem = torch.linalg.norm(numerator)
    return numerator/dem 

def get_W(xc, Tlw, nodes, kdtree):
    dqb = blending(xc, nodes, kdtree).view(1,8)
    T = Tlw @ SE3(dqb)
    return T 

def test_get_W():
    t1 = time.time()
    subsample_rate = 10
    human = trimesh.load('reconstruct/seq006/mesh.ply')
    verts, faces = human.vertices, human.faces 
    verts = torch.tensor(verts.copy())
    faces = torch.tensor(faces.copy())
    average_distances = []
    for f in faces:
        average_distances.append(average_edge_dist_in_face(f, verts))

    t2 = time.time()
    print("load mesh: ", t2-t1 )
    _radius = subsample_rate * torch.mean(torch.tensor(average_distances))
    # print(verts.shape, faces.shape, _radius)
    _vertices = verts 
    _faces = faces
    t3 = time.time()
    print("sub * average ", t3 - t2)
    nodes_v, nodes_idx = uniform_sample(_vertices, _radius)
    nodes_v_t = torch.tensor(nodes_v)
    assert nodes_v_t.shape[1] == 3
    t4 = time.time()
    print("uniform sample ", t4-t3)
    # define list of node warp
    _nodes = []  
    for j in range(len(nodes_v)):
        _nodes.append([nodes_idx[j],
                            nodes_v_t[j],
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # real Tic = T_k-1
                            torch.tensor([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00]).float(), # tmp T_k for optim
                            2 * _radius])
    t5 = time.time()
    print("append nodes ", t5-t4)
    # construct kd tree
    _kdtree = KDTree(nodes_v)
    t6 = time.time() 
    print("kd tree ", t6-t5)
    # cache all neighbor node for all vertices in canonical space
    _neighbor_look_up = []
    for vert in verts:
        dists, idx = _kdtree.query(vert, k=4)
        _neighbor_look_up.append(idx) 
    t7 = time.time()
    print("Add look up ", t7 - t6)
    xc = torch.tensor([[1,1,1]]).float()
    Tlw = torch.eye(4)
    # def energy(xc, Tlw, _nodes, _kdtree) 
    # khong duoc, phai viet get_w theo kieu truyen thang neighbour luon
    # theo toan thi moi W là 1 xc + bo neighbors, nen neu batch xc thi + batch bo neighbor. Vectorize duoc. 
    # xc [bs, 3], neighbor [bs, 8], [bs,3] and [bs, 1]. Chi argnum = bs,8 de lay jac. sau khi tinh xong thi concate ket qua vo jac cuoi cung hoac GN luon neu muon 
    T = get_W(xc, Tlw, _nodes, _kdtree)
    R, t= decompose_se3(T)
    xt =  (R@ xc.T.reshape(3,1) + t).T.view(xc.shape)
    
    assert xt.shape == (1,3)