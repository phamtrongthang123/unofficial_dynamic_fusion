import numpy as np 
from numpy import linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import torch 
import pytorch3d
import pytorch3d.utils
from pytorch3d.io import IO, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, HardPhongShader
)
from pytorch3d.ops import interpolate_face_attributes

# Radius-based spatial subsampling
def uniform_sample(arr,radius):
    """
    arr: (N,3) vertices ndarray
    radius: float radius = subsample_rate * np.average(np.array(average_distances))
    """
    candidates = np.array(arr).copy()
    locations = np.arange(len(candidates))

    result = []
    result_idx = []
    
    while candidates.size > 0:
        remove = []
        rows = len(candidates)
        sample = candidates[0]
        index = np.arange(rows).reshape(-1,1)
        dists = np.column_stack((index,candidates))
        result.append(sample)
        result_idx.append(locations[0])
        for row in dists:
            if la.norm(row[1:] - sample) < radius:
                remove.append(int(row[0]))
        candidates = np.delete(candidates, remove, axis=0)
        locations = np.delete(locations, remove)
    return np.array(result), np.array(result_idx)

def cal_dist(a,b):
    return la.norm(a-b)



def decompose_se3(M):
    return M[:, :3,:3], M[:, :3, [3]] # (torch.Size([bs, 3, 3]), torch.Size([bs, 3, 1]))

  
def get_diag(a):
    return a.diagonal(dim1=-2, dim2=-1).diag_embed(dim1=-2, dim2=-1)

def cat1(a):
    assert len(a.shape) == 2 # batch, 3
    return torch.cat([a, torch.ones(a.shape[0],1)], dim=1)

def custom_transpose(a):
    return a.permute(*torch.arange(a.ndim - 1, -1, -1))
def custom_transpose_batch(a, isknn=False):
    if isknn:
        return a.permute(0,1,*torch.arange(a.ndim - 1, 1, -1))
    return a.permute(0,*torch.arange(a.ndim - 1, 0, -1))

def batch_eye(bs, dim):
    x = torch.eye(dim)
    x = x.reshape((1, dim, dim))
    y = x.repeat(bs, 1, 1)
    return y
def compose_se3(R,t):
    bs, _,_ = R.shape
    T = batch_eye(bs, 4)
    T[:,:3,:3] = R.clone()
    T[:,:3,[3]] = t.clone()
    return T

def robust_Tukey_penalty(value, c):
  # https://mathworld.wolfram.com/TukeysBiweight.html
  if value.abs() > c: 
    return torch.tensor(0.0) 
  return value * (1- value**2 / c**2)**2

def huber_penalty(value,c):
    if value.abs() <= c:
        return 0.5 * (value**2)
    else:
        return c * (value.abs() - 0.5*c)

def SE3(dq):
  # from https://www.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
  # i have to do this because vmap require outplace operator
  out = []
  dq = dq.view(1,8)
  out.append(1 - 2*(dq[:,2]**2 + dq[:,3]**2))
  out.append(2 * dq[:,1]*dq[:,2] - 2*dq[:,0]*dq[:,3])
  out.append(2 * dq[:,1]*dq[:,3] + 2*dq[:,0]*dq[:,2] )
  out.append(2 * (-dq[:,4]*dq[:,1] + dq[:,5]*dq[:,0] - dq[:,6] *dq[:,3] + dq[:,7]*dq[:,2] ))
  out.append(2 *dq[:,1]*dq[:,2] + 2*dq[:,0]*dq[:,3] )
  out.append(1 - 2*dq[:,1]**2 - 2*dq[:,3]**2 )
  out.append(2*dq[:,2]*dq[:,3] - 2*dq[:,0]*dq[:,1]) 
  out.append(2*(-dq[:,4]*dq[:,2] + dq[:,5]*dq[:,3] + dq[:,6]*dq[:,0] - dq[:,7]*dq[:,1]))
  out.append(2*dq[:,1]*dq[:,3] - 2*dq[:,0]*dq[:,2] )
  out.append(2*dq[:,2]*dq[:,3] + 2 *dq[:,0]*dq[:,1] )
  out.append(1-2*dq[:,1]**2 - 2*dq[:,2]**2 )
  out.append(2*(-dq[:,4]*dq[:,3] -dq[:,5]*dq[:,2] +dq[:,6]*dq[:,1] +dq[:,7]*dq[:,0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([1]))
  return torch.cat(out).view(1,4,4)

def SE3_novmap(dq):
  # from https://www.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
  # add if else inside a function with tracing is asking for trouble, so I split another function for quick testing
  # this dq shape = bs,knn,8
  out = []
  out.append(1 - 2*(dq[:,2]**2 + dq[:,3]**2))
  out.append(2 * dq[:,1]*dq[:,2] - 2*dq[:,0]*dq[:,3])
  out.append(2 * dq[:,1]*dq[:,3] + 2*dq[:,0]*dq[:,2] )
  out.append(2 * (-dq[:,4]*dq[:,1] + dq[:,5]*dq[:,0] - dq[:,6] *dq[:,3] + dq[:,7]*dq[:,2] ))
  out.append(2 *dq[:,1]*dq[:,2] + 2*dq[:,0]*dq[:,3] )
  out.append(1 - 2*dq[:,1]**2 - 2*dq[:,3]**2 )
  out.append(2*dq[:,2]*dq[:,3] - 2*dq[:,0]*dq[:,1]) 
  out.append(2*(-dq[:,4]*dq[:,2] + dq[:,5]*dq[:,3] + dq[:,6]*dq[:,0] - dq[:,7]*dq[:,1]))
  out.append(2*dq[:,1]*dq[:,3] - 2*dq[:,0]*dq[:,2] )
  out.append(2*dq[:,2]*dq[:,3] + 2 *dq[:,0]*dq[:,1] )
  out.append(1-2*dq[:,1]**2 - 2*dq[:,2]**2 )
  out.append(2*(-dq[:,4]*dq[:,3] -dq[:,5]*dq[:,2] +dq[:,6]*dq[:,1] +dq[:,7]*dq[:,0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([0]))
  out.append(torch.tensor([1]))
  return torch.cat(out).view(dq.shape[0],4,4)

def average_edge_dist_in_face( f, verts):
    v1 = verts[f[0]]
    v2 = verts[f[1]]
    v3 = verts[f[2]]
    return (cal_dist(v1,v2) + cal_dist(v1,v3) + cal_dist(v2,v3))/3


def blending(Xc, dqs, dgw, dgv):
    repxc = Xc.view(1, 3).repeat_interleave(dgv.shape[0],0)
    assert repxc.shape == dgv.shape 
    wixc = torch.exp(-torch.linalg.norm(dgv - repxc, dim=1)**2 / (2*torch.pow(dgw,2)))
    assert wixc.shape == dgw.shape
    qkc = torch.einsum('i,ik->k',wixc,dqs)
    # qkc = dqs[0]+dqs[1]
    assert qkc.shape[0] == 8 and len(qkc.shape)==1
    # dem = torch.linalg.norm(qkc[:4])
    out = dqnorm(qkc)
    # print(out)
    return out 

def dqnorm(dq):
    norm = torch.linalg.norm(dq[:4].view(-1))
    dq1 = dq[:4]/norm 
    dq2 = dq[4:]/norm 
    dq2 = dq2 - torch.dot(dq2, dq1) * dq1
    dq_ret = torch.cat((dq1, dq2))
    return dq_ret

def get_W(Xc, Tlw, dqs, dgw, dgv):
    dqb = blending(Xc, dqs, dgw, dgv).view(1,8)
    T = torch.einsum('ij,bjk -> bik',Tlw, SE3(dqb))
    return T 


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