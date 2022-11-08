import numpy as np
from numpy import linalg as la
import torch
import pytorch3d
import pytorch3d.utils
from pytorch3d.structures import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer
from pytorch3d.ops import interpolate_face_attributes
from functorch import vmap
import time
import matplotlib.pyplot as plt
import os
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import plotly.express as px
import torch 

# Radius-based spatial subsampling
def uniform_sample(arr: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
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
        index = np.arange(rows).reshape(-1, 1)
        dists = np.column_stack((index, candidates))
        result.append(sample)
        result_idx.append(locations[0])
        for row in dists:
            if la.norm(row[1:] - sample) < radius:
                remove.append(int(row[0]))
        candidates = np.delete(candidates, remove, axis=0)
        locations = np.delete(locations, remove)
    return np.array(result), np.array(result_idx)


def cal_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return la.norm(a - b)


def decompose_se3(M: torch.tensor) -> torch.tensor:
    # (torch.Size([bs, 3, 3]), torch.Size([bs, 3, 1]))
    return M[:, :3, :3], M[:, :3, [3]]


def get_diag(a: torch.tensor) -> torch.tensor:
    return a.diagonal(dim1=-2, dim2=-1).diag_embed(dim1=-2, dim2=-1)


def cat1(a: torch.tensor) -> torch.tensor:
    assert len(a.shape) == 2  # batch, 3
    return torch.cat([a, torch.ones(a.shape[0], 1)], dim=1)


def custom_transpose(a: torch.tensor) -> torch.tensor:
    return a.permute(*torch.arange(a.ndim - 1, -1, -1))


def custom_transpose_batch(a: torch.tensor, isknn: bool = False) -> torch.tensor:
    if isknn:
        return a.permute(0, 1, *torch.arange(a.ndim - 1, 1, -1))
    return a.permute(0, *torch.arange(a.ndim - 1, 0, -1))


def batch_eye(
    bs: Union[torch.tensor, int], dim: Union[torch.tensor, int]
) -> torch.tensor:
    x = torch.eye(dim)
    x = x.reshape((1, dim, dim))
    y = x.repeat(bs, 1, 1)
    return y


def compose_se3(R: torch.tensor, t: torch.tensor) -> torch.tensor:
    bs, _, _ = R.shape
    T = batch_eye(bs, 4)
    T[:, :3, :3] = R.clone()
    T[:, :3, [3]] = t.clone()
    return T


def robust_Tukey_penalty(value: torch.tensor, c: torch.tensor) -> torch.tensor:
    # https://mathworld.wolfram.com/TukeysBiweight.html
    ae = torch.abs(value)
    return torch.where(ae > c, torch.tensor(0.0).float(), value * (1 - value**2 / c**2) ** 2).type_as(
        value
    )


def huber_penalty(value, c):
    ae = torch.abs(value)
    return torch.where(ae <= c, 0.5 * (value**2), c * (ae - 0.5 * c)).type_as(value)


def SE3(dq: torch.tensor) -> torch.tensor:
    # from https://www.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
    # i have to do this because vmap require outplace operator
    out = []
    dq = dq.view(1, 8)
    out.append(1 - 2 * (dq[:, 2] ** 2 + dq[:, 3] ** 2))
    out.append(2 * dq[:, 1] * dq[:, 2] - 2 * dq[:, 0] * dq[:, 3])
    out.append(2 * dq[:, 1] * dq[:, 3] + 2 * dq[:, 0] * dq[:, 2])
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 1]
            + dq[:, 5] * dq[:, 0]
            - dq[:, 6] * dq[:, 3]
            + dq[:, 7] * dq[:, 2]
        )
    )
    out.append(2 * dq[:, 1] * dq[:, 2] + 2 * dq[:, 0] * dq[:, 3])
    out.append(1 - 2 * dq[:, 1] ** 2 - 2 * dq[:, 3] ** 2)
    out.append(2 * dq[:, 2] * dq[:, 3] - 2 * dq[:, 0] * dq[:, 1])
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 2]
            + dq[:, 5] * dq[:, 3]
            + dq[:, 6] * dq[:, 0]
            - dq[:, 7] * dq[:, 1]
        )
    )
    out.append(2 * dq[:, 1] * dq[:, 3] - 2 * dq[:, 0] * dq[:, 2])
    out.append(2 * dq[:, 2] * dq[:, 3] + 2 * dq[:, 0] * dq[:, 1])
    out.append(1 - 2 * dq[:, 1] ** 2 - 2 * dq[:, 2] ** 2)
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 3]
            - dq[:, 5] * dq[:, 2]
            + dq[:, 6] * dq[:, 1]
            + dq[:, 7] * dq[:, 0]
        )
    )
    out.append(torch.tensor([0]).type_as(out[-1]))
    out.append(torch.tensor([0]).type_as(out[-1]))
    out.append(torch.tensor([0]).type_as(out[-1]))
    out.append(torch.tensor([1]).type_as(out[-1]))
    return torch.cat(out).view(1, 4, 4)


def SE3_novmap(dq: torch.tensor) -> torch.tensor:
    # from https://www.cs.utah.edu/~ladislav/kavan07skinning/kavan07skinning.pdf
    # add if else inside a function with tracing is asking for trouble, so I split another function for quick testing
    # this dq shape = bs,knn,8
    out = []
    out.append(1 - 2 * (dq[:, 2] ** 2 + dq[:, 3] ** 2))
    out.append(2 * dq[:, 1] * dq[:, 2] - 2 * dq[:, 0] * dq[:, 3])
    out.append(2 * dq[:, 1] * dq[:, 3] + 2 * dq[:, 0] * dq[:, 2])
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 1]
            + dq[:, 5] * dq[:, 0]
            - dq[:, 6] * dq[:, 3]
            + dq[:, 7] * dq[:, 2]
        )
    )
    out.append(2 * dq[:, 1] * dq[:, 2] + 2 * dq[:, 0] * dq[:, 3])
    out.append(1 - 2 * dq[:, 1] ** 2 - 2 * dq[:, 3] ** 2)
    out.append(2 * dq[:, 2] * dq[:, 3] - 2 * dq[:, 0] * dq[:, 1])
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 2]
            + dq[:, 5] * dq[:, 3]
            + dq[:, 6] * dq[:, 0]
            - dq[:, 7] * dq[:, 1]
        )
    )
    out.append(2 * dq[:, 1] * dq[:, 3] - 2 * dq[:, 0] * dq[:, 2])
    out.append(2 * dq[:, 2] * dq[:, 3] + 2 * dq[:, 0] * dq[:, 1])
    out.append(1 - 2 * dq[:, 1] ** 2 - 2 * dq[:, 2] ** 2)
    out.append(
        2
        * (
            -dq[:, 4] * dq[:, 3]
            - dq[:, 5] * dq[:, 2]
            + dq[:, 6] * dq[:, 1]
            + dq[:, 7] * dq[:, 0]
        )
    )
    out.append(torch.tensor([0]))
    out.append(torch.tensor([0]))
    out.append(torch.tensor([0]))
    out.append(torch.tensor([1]))
    return torch.cat(out).view(dq.shape[0], 4, 4)


def average_edge_dist_in_face(f: np.ndarray, verts: np.ndarray) -> np.ndarray:
    v1 = verts[f[0]]
    v2 = verts[f[1]]
    v3 = verts[f[2]]
    return (cal_dist(v1, v2) + cal_dist(v1, v3) + cal_dist(v2, v3)) / 3


def blending(
    Xc: torch.tensor, dqs: torch.tensor, dgw: torch.tensor, dgv: torch.tensor
) -> torch.tensor:
    """_summary_

    Args:
        Xc (torch.tensor): _description_
        dqs (torch.tensor): _description_
        dgw (torch.tensor): _description_
        dgv (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """
    repxc = Xc.view(1, 3).repeat_interleave(dgv.shape[0], 0).type_as(dgv)
    assert repxc.shape == dgv.shape
    wixc = torch.exp(
        -torch.linalg.norm(dgv - repxc, dim=1) ** 2 / (2 * torch.pow(dgw, 2))
    ).float()
    assert wixc.shape == dgw.shape
    qkc = torch.einsum("i,ik->k", wixc, dqs)
    # qkc = dqs[0]+dqs[1]
    assert qkc.shape[0] == 8 and len(qkc.shape) == 1
    # dem = torch.linalg.norm(qkc[:4])
    out = dqnorm(qkc)
    # print(out)
    return out


def dqnorm(dq: torch.tensor) -> torch.tensor:
    """_summary_

    Args:
        dq (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """    
    norm = torch.linalg.norm(dq[:4].view(-1))
    dq1 = dq[:4] / norm
    dq2 = dq[4:] / norm
    dq2 = dq2 - torch.dot(dq2, dq1) * dq1
    dq_ret = torch.cat((dq1, dq2))
    return dq_ret


def get_W(
    Xc: torch.tensor,
    Tlw: torch.tensor,
    dqs: torch.tensor,
    dgw: torch.tensor,
    dgv: torch.tensor,
) -> torch.tensor:
    """_summary_

    Args:
        Xc (torch.tensor): _description_
        Tlw (torch.tensor): _description_
        dqs (torch.tensor): _description_
        dgw (torch.tensor): _description_
        dgv (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """
    dqb = blending(Xc, dqs, dgw, dgv).view(1, 8)
    T = torch.einsum("ij,bjk -> bik", Tlw, SE3(dqb))
    return T


def render_depth(
    R: torch.tensor,
    t: torch.tensor,
    K: torch.tensor,
    verts: torch.tensor,
    faces: torch.tensor,
    image_size: Tuple[int, int],
    device: Union[str, Any],
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """_summary_

    Args:
        R (torch.tensor): _description_
        t (torch.tensor): _description_
        K (torch.tensor): _description_
        verts (torch.tensor): _description_
        faces (torch.tensor): _description_
        image_size (Tuple[int, int]): _description_
        device (Union[str, Any]): _description_

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: _description_
    """
    assert K.shape == (1, 3, 3), print(K.shape)
    assert R.shape == (1, 3, 3), print(R.shape)
    assert t.shape == (1, 3), print(t, t.shape)
    verts = verts.to(device)
    faces = faces.to(device)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        faces_per_pixel=1,
        bin_size=None,
    )
    mesh = Meshes(verts=[verts], faces=[faces])
    image_size_t = torch.tensor(image_size).view(1, 2)
    camera_torch = pytorch3d.utils.cameras_from_opencv_projection(
        R=R, tvec=t, camera_matrix=K, image_size=image_size_t
    )
    mesh_raster = MeshRasterizer(
        cameras=camera_torch, raster_settings=raster_settings
    ).to(device)
    fragments = mesh_raster(mesh)
    depth_map = fragments.zbuf[0].view(image_size)
    vertex_normals = mesh.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    # normal map don't need interpolate bary coord
    normal_map = interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    ).view(image_size[0], image_size[1], 3)
    # vertex needs it though
    vertex_map = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, verts[faces]
    ).view(image_size[0], image_size[1], 3)
    return depth_map, normal_map, vertex_map


def warp_helper(
    Xc: torch.tensor,
    Tlw: torch.tensor,
    dgv: torch.tensor,
    dgse: torch.tensor,
    dgw: torch.tensor,
    node_to_nn: torch.tensor,
) -> torch.tensor:
    """This function will warp the Xc from canonical space into live frame for you.
    Please wrap this function inside a vmap before using.

    Args:
        Xc (torch.tensor): _description_
        Tlw (torch.tensor): _description_
        dgv (torch.tensor): _description_
        dgse (torch.tensor): _description_
        dgw (torch.tensor): _description_
        node_to_nn (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """
    dgv_nn = dgv[node_to_nn]
    dgw_nn = dgw[node_to_nn]
    dgse_nn = dgse[node_to_nn]
    T = get_W(Xc, Tlw, dgse_nn, dgw_nn, dgv_nn)
    R, t = decompose_se3(T.type_as(Xc))
    Xt = (
        torch.einsum("bij,bjk->bi", R, Xc.view(1, 3, 1)) + t.squeeze(-1)
    ).squeeze()  # shape [3] == target

    return Xt


def warp_to_live_frame(
    verts: torch.tensor,
    Tlw: torch.tensor,
    dgv: torch.tensor,
    dgse: torch.tensor,
    dgw: torch.tensor,
    kdtree: Any,
    device: str = "cpu",
) -> torch.tensor:
    assert len(verts.shape) == 2 and verts.shape[1] == 3
    knn = 4
    t1 = time.time()
    dists, idx = kdtree.query(verts, k=knn, workers=-1)
    node_to_nn = torch.tensor(idx).to(device).long()
    t2 = time.time()
    print("find neighbor cost: ", t2 - t1)
    verts = verts.to(device)
    Tlw, dgv, dgse, dgw = (
        Tlw.to(device),
        dgv.to(device),
        dgse.to(device),
        dgw.to(device),
    )
    vmap_helper = vmap(warp_helper, in_dims=(0, None, None, None, None, 0))
    verts_live = vmap_helper(verts, Tlw, dgv, dgse, dgw, node_to_nn)
    print("warping cost: ", time.time() - t2)
    assert len(verts_live.shape) == 2 and verts_live.shape[1] == 3
    return verts_live.to(device)


def plot_vis_depthmap(depth_map: torch.tensor, vis_dir: str, i: int) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    plt.figure()
    plt.imshow(depth_map.detach().cpu())  # (h,w)
    plt.colorbar(label="Distance to Camera")
    plt.title("Depth image")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.savefig(f"{vis_dir}/{str(i).zfill(6)}.jpg")
    plt.close()


def plot_heatmap_step(a: torch.tensor, vis_dir: str, index: int, step: int) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    b= a.abs().sum(dim=0).cpu()

    fig = px.imshow(b, text_auto=True)
    fig.write_image(f"{vis_dir}/{str(index).zfill(6)}_{str(step).zfill(3)}.png", scale=6, width=700, height=700)

# the translation part has some problem when convert back and forth, I will need to investigate it
def quat_from_rot(_rot: torch.tensor) -> torch.tensor:

    tr = _rot[0, 0] + _rot[1, 1] + _rot[2, 2]
    ifhere = (
        torch.stack(
            (
                tr > 0,
                torch.logical_and(
                    _rot[0, 0] > _rot[1, 1], _rot[0, 0] > _rot[2, 2]
                ).type_as(_rot),
                _rot[1, 1] > _rot[2, 2],
                torch.tensor(True).type_as(_rot),
            )
        )
        .long()
        .view(-1)
    )
    s = torch.sqrt(tr + 1.0) * 2
    tmp1 = []
    tmp1.append(s * 0.25)
    tmp1.append((_rot[2, 1] - _rot[1, 2]) / s)
    tmp1.append((_rot[0, 2] - _rot[2, 0]) / s)
    tmp1.append((_rot[1, 0] - _rot[0, 1]) / s)
    re = torch.stack(tmp1).view(1, 4)
    s = torch.sqrt(1.0 + _rot[0, 0] - _rot[1, 1] - _rot[2, 2]) * 2
    w = (_rot[2, 1] - _rot[1, 2]) / s
    x = 0.25 * s
    y = (_rot[0, 1] + _rot[1, 0]) / s
    z = (_rot[0, 2] + _rot[2, 0]) / s
    re = torch.cat((re, torch.stack((w, x, y, z)).view(1, 4)), dim=0)
    s = torch.sqrt(1.0 + _rot[1, 1] - _rot[0, 0] - _rot[2, 2]) * 2
    w = (_rot[0, 2] - _rot[2, 0]) / s
    x = (_rot[0, 1] + _rot[1, 0]) / s
    y = 0.25 * s
    z = (_rot[1, 2] + _rot[2, 1]) / s
    re = torch.cat((re, torch.stack((w, x, y, z)).view(1, 4)), dim=0)
    s = torch.sqrt(1.0 + _rot[2, 2] - _rot[0, 0] - _rot[1, 1]) * 2
    w = (_rot[1, 0] - _rot[0, 1]) / s
    x = (_rot[0, 2] + _rot[2, 0]) / s
    y = (_rot[1, 2] + _rot[2, 1]) / s
    z = 0.25 * s
    re = torch.cat((re, torch.stack((w, x, y, z)).view(1, 4)), dim=0).type_as(_rot)
    # I must write it this way so vmap can accept it as "if else", can't do re[ifhere] anyway.
    re_withoutinf = torch.where(
        torch.any(torch.isinf(re), dim=1).view(-1, 1),
        torch.tensor([1, 0, 0, 0]).type_as(_rot),
        re,
    )
    re_withoutnan = torch.where(
        torch.any(torch.isnan(re_withoutinf), dim=1).view(-1, 1),
        torch.tensor([1, 0, 0, 0]).type_as(_rot),
        re_withoutinf,
    )
    ree = torch.matmul(
        (torch.logical_and(ifhere, torch.cumsum(ifhere, dim=0) == 1)).float(),
        re_withoutnan,
    )
    return ree


def quat_add(q0: torch.tensor, q1: torch.tensor) -> torch.tensor:
    return q0 + q1


def quat_mul(_q0: torch.tensor, _q1: torch.tensor) -> torch.tensor:
    w = _q0[0] * _q1[0] - _q0[1] * _q1[1] - _q0[2] * _q1[2] - _q0[3] * _q1[3]
    x = _q0[0] * _q1[1] + _q0[1] * _q1[0] + _q0[2] * _q1[3] - _q0[3] * _q1[2]
    y = _q0[0] * _q1[2] - _q0[1] * _q1[3] + _q0[2] * _q1[0] + _q0[3] * _q1[1]
    z = _q0[0] * _q1[3] + _q0[1] * _q1[2] - _q0[2] * _q1[1] + _q0[3] * _q1[0]
    return torch.stack((w, x, y, z))


def dual_quat_mul(q0: torch.tensor, q1: torch.tensor) -> torch.tensor:
    quat0 = quat_mul(q0[:4], q1[:4])
    quat1 = quat_add(quat_mul(q0[:4], q1[4:]), quat_mul(q0[4:], q1[:4]))
    return torch.cat((quat0, quat1))


def SE3_dq(mat44: torch.tensor) -> torch.tensor:
    R = mat44[:3, :3]
    t = mat44[:3, 3]
    rot_part = torch.cat((quat_from_rot(R), torch.zeros(4).type_as(R)))
    # i have to do this because of vmap
    vec_part = (
        torch.stack(
            (
                torch.tensor(1).type_as(R),
                torch.tensor(0).type_as(R),
                torch.tensor(0).type_as(R),
                torch.tensor(0).type_as(R),
                torch.tensor(0).type_as(R),
                0.5 * t[0],
                0.5 * t[1],
                0.5 * t[2],
            )
        )
        .type_as(R)
        .view(-1)
    )
    return dual_quat_mul(vec_part, rot_part)


def quat_conj(quat: torch.tensor) -> torch.tensor:
    return torch.tensor([quat[0], -quat[1], -quat[2], -quat[3]])


def mat34(dq: torch.tensor) -> torch.tensor:
    dq_norm = dqnorm(dq)
    r = SE3(dq).view(4, 4)[:3, :3]
    vec_part = 2.0 * quat_mul(dq_norm[4:], quat_conj(dq_norm[:4]))
    t = vec_part[1:]
    print(t.shape, r.shape)

    return torch.cat(
        (torch.cat((r, t.view(3, 1)), dim=1), torch.tensor([0, 0, 0, 1]).view(1, 4)),
        dim=0,
    )
