import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import torch
from scipy.spatial import KDTree
from numpy import linalg as la
import time 
from tmp_utils import SE3, decompose_se3, blending, get_diag, dqnorm, custom_transpose_batch, get_W, render_depth, plot_heatmap_step, warp_helper, robust_Tukey_penalty, SE3_dq
import numpy as np 
from functorch import vmap, vjp, jacrev, jacfwd, hessian, jvp, grad
from einops import rearrange, reduce, repeat
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import List, Set, Dict, Tuple, Optional, Union, Any

def plot_xc_xt_nt(Xc, Xt, nt, volume_live, vis_dir: str, step: int) -> None:
    os.makedirs(vis_dir, exist_ok=True)
    fig = ff.create_quiver(Xt[:,0], Xt[:,1], nt[:,0], nt[:,1],
                        scale=0.025*1.,
                        arrow_scale=.2*0.025,
                        name='quiver',
                        line_width=1)
    fig.add_trace(go.Scatter(x=Xc[:,0], y=Xc[:,1], mode='markers+lines'))
    fig.add_trace(go.Scatter(x=Xt[:,0], y=Xt[:,1], mode='markers+lines'))
    fig.add_trace(go.Scatter(x=volume_live[:,0], y=volume_live[:,1], mode='markers'))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    # fig.show()
    fig.write_image(f"{vis_dir}/{str(step).zfill(3)}.png", scale=6, width=500, height=500)
def produce_2d_normal_rightside(a: torch.tensor) -> torch.tensor:
    # produce a normal vector that always point to the right side 
    assert len(a.shape) == 2 
    assert a.shape[1] == 2
    res = []
    for i in range(a.shape[0]):
        if i == 0: 
            tmp = sorted([a[i], a[i+1]], key=lambda x: x[1])
            n = (tmp[0] - tmp[1]).flip(dims=(0,))
            n[0] = -n[0]
        elif i == a.shape[0] - 1:
            tmp = sorted([a[i], a[i-1]], key=lambda x: x[1])
            n = (tmp[0] - tmp[1]).flip(dims=(0,))
            n[0] = -n[0]
        else:
            tmp1 = sorted([a[i], a[i+1]], key=lambda x: x[1])
            tmp2 = sorted([a[i-1], a[i]], key=lambda x: x[1])
            n = (((tmp2[0] - tmp2[1])+(tmp1[0] - tmp1[1]))/2).flip(dims=(0,))
            n[0] = -n[0]
        res.append(n/n.norm())
    return torch.stack(res)

def cat_z0(a: torch.tensor) -> torch.tensor:
    return torch.cat([a, torch.zeros(a.shape[0],1)], dim=1)

def init_dgse_dgw(dgv: torch.tensor, _radius: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    dgse = []  
    dgw = []
    for j in range(len(dgv)):
        dgse.append(torch.tensor([ 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]).float())
        dgw.append(torch.tensor(3.0*_radius))
    dgse = torch.stack(dgse)
    dgw = torch.stack(dgw)
    return dgse, dgw

def make_node_nn(Xc: torch.tensor, _kdtree: Any, knn: int ) -> torch.tensor:
    node_to_nn = []
    for i in range(Xc.shape[0]):
        dists, idx = _kdtree.query(Xc[i], k=knn)
        node_to_nn.append(torch.tensor(idx))
    node_to_nn = torch.stack(node_to_nn)
    return node_to_nn

def solve_(A: torch.tensor,b: torch.tensor) -> torch.tensor:
    solved_delta = torch.linalg.lstsq(A.view(6,6), b.view(6,1))
    res = solved_delta.solution 
    return res




def reg_term(
    Tlw: torch.tensor,
    dgv: torch.tensor,
    dgse: torch.tensor,
    dgw: torch.tensor,
    dgv_nn: torch.tensor,
) -> torch.tensor:
    """Regularization term. This method computes the as-rigid-as-possible energy described in the dynamic fusion paper.
    However, the original equation uses sum only, which potentially causes exploding gradient, so I use mean instead of sum. Note that the scalar \alpha_ij is merged outside. In the paper, it is 200. With my setting is 0.025 radius, for example, so the weight outside is 5.

    Eq (8): mean(T_ic dg_v^j - T_jc dg_v^j)

    Args:
        Tlw (torch.tensor)      : Shape (4,4).  The explicit rigid transformation.
        dgv (torch.tensor)      : Shape (N,3).  The nodes position.
        dgse (torch.tensor)     : Shape (N,8).  The se3 transformation represented as dual quaternion of the nodes.
        dgw (torch.tensor)      : Shape (N,1).  The influential radius of the nodes.
        dgv_nn (torch.tensor)   : Shape (N,4).  The index list of the nodes' neighbors

    Returns:
        torch.tensor:           : Shape (1).    Return the scalar result from the mentioned Eq.
    """
    shape = dgv.shape
    knn = dgv_nn.shape[1]
    # self warp
    warp_helper_vmap = vmap(warp_helper, in_dims=(0, None, None, None, None, 0))
    dgv_warped = warp_helper_vmap(dgv, Tlw, dgv, dgse, dgw, dgv_nn)

    # self wapr with neighbors' se3
    dgv_nn_nn = dgv_nn[dgv_nn].view(-1, knn)
    dgv_rep = dgv.view(-1, 1, 3).repeat_interleave(knn, 1).view(-1, 3)
    dgv_nn_warp = warp_helper_vmap(dgv_rep, Tlw, dgv, dgse, dgw, dgv_nn_nn).view(
        -1, knn, 3
    )
    dgv_warped_rep = dgv_warped.view(-1, 1, 3).repeat_interleave(knn, 1)
    el2 = torch.linalg.norm(dgv_nn_warp - dgv_warped_rep, dim=2) ** 2
    ssel2 = el2.mean()
    return ssel2

def make_q0(theta: torch.tensor) -> torch.tensor:
    return torch.cos(torch.norm(theta))
def make_q1(theta: torch.tensor) -> torch.tensor:
    return theta[0]*torch.sinc(torch.norm(theta))
def make_q2(theta: torch.tensor) -> torch.tensor:
    return theta[1]*torch.sinc(torch.norm(theta))
def make_q3(theta: torch.tensor) -> torch.tensor:
    return theta[2]*torch.sinc(torch.norm(theta))

def make_q4(theta: torch.tensor, t: torch.tensor) -> torch.tensor:
    return -1/2 * (t[0]*make_q1(theta) + t[1]*make_q2(theta) + t[2]*make_q3(theta))

def make_q5(theta: torch.tensor, t: torch.tensor) -> torch.tensor:
    return 1/2 * (t[0]*make_q0(theta) + t[1]*make_q3(theta) - t[2]*make_q2(theta))
def make_q6(theta: torch.tensor, t: torch.tensor) -> torch.tensor:
    return 1/2 * (-t[0]*make_q3(theta) + t[1]*make_q0(theta) + t[2]*make_q1(theta))
def make_q7(theta,t: torch.tensor) -> torch.tensor:
    return 1/2 * (t[0]*make_q2(theta)- t[1]*make_q1(theta) + t[2]*make_q0(theta))
def make_dq(theta, t: torch.tensor) -> torch.tensor:
    return torch.stack([make_q0(theta),make_q1(theta), make_q2(theta), make_q3(theta),make_q4(theta,t) , make_q5(theta,t), make_q6(theta,t), make_q7(theta,t)])

def make_dq_from_vec(vec: torch.tensor) -> torch.tensor:
    theta = vec[:3]
    t = vec[3:]
    return make_dq(theta, t)

def energy(
    gt_Xc: torch.tensor,
    Tlw: torch.tensor,
    dgv: torch.tensor,
    dgse: torch.tensor,
    dgw: torch.tensor,
    node_to_nn: torch.tensor,
    dgv_nn: torch.tensor,
) -> Tuple[torch.tensor, torch.tensor]:
    """_summary_

    Args:
        gt_Xc (torch.tensor): _description_
        Tlw (torch.tensor): _description_
        dgv (torch.tensor): _description_
        dgse (torch.tensor): _description_
        dgw (torch.tensor): _description_
        node_to_nn (torch.tensor): _description_
        dgv_nn (torch.tensor): _description_

    Returns:
        Tuple[torch.tensor, torch.tensor]: _description_
    """
    data_vmap = vmap(data_term, in_dims=(0, None, None, None, None, 0))
    make_dq_from_vec_vmap = vmap(make_dq_from_vec, in_dims=(0))
    dgse_dq = make_dq_from_vec_vmap(dgse)
    data_val = data_vmap(gt_Xc, Tlw, dgv, dgse_dq, dgw, node_to_nn).mean()
    reg_val = reg_term(Tlw, dgv, dgse_dq, dgw, dgv_nn)
    # re = ((data_val + 5*reg_val) / 2).float() # 5 = lambda in surfelwarp paper
    re = data_val
    return re, re

def post_process_solved_delta(solved_delta: Any, bs: int, dgv: torch.tensor, dgse: torch.tensor) -> torch.tensor:
    try:
        solved_delta = solved_delta.solution.view(bs, len(dgv), 6).mean(dim=0)
    except: 
        solved_delta = solved_delta.view(bs, len(dgv), 6).mean(dim=0)
    # eliminate nan and inf
    solved_delta = torch.where(
        torch.any(torch.isnan(solved_delta.view(dgse.shape))).view(-1, 1),
        torch.zeros(6).type_as(dgse),
        solved_delta.view(dgse.shape),
    )
    solved_delta = torch.where(
        torch.any(torch.isinf(solved_delta.view(dgse.shape))).view(-1, 1),
        torch.zeros(6).type_as(dgse),
        solved_delta.view(dgse.shape),
    )
    return solved_delta

def data_term(
    gt_Xc: torch.tensor,
    Tlw: torch.tensor,
    dgv: torch.tensor,
    dgse: torch.tensor,
    dgw: torch.tensor,
    node_to_nn: torch.tensor,
) -> torch.tensor:
    """_summary_

    Args:
        gt_Xc (torch.tensor): _description_
        Tlw (torch.tensor): _description_
        dgv (torch.tensor): _description_
        dgse (torch.tensor): _description_
        dgw (torch.tensor): _description_
        node_to_nn (torch.tensor): _description_

    Returns:
        torch.tensor: _description_
    """
    Xc = gt_Xc[2]
    nc = gt_Xc[3]
    gt_v = gt_Xc[0]
    gt_n = gt_Xc[1]
    Xt = warp_helper(Xc, Tlw, dgv, dgse, dgw, node_to_nn)
    # if you want it to similar to surfel warp, uncomment the lines below
    alpha = 1.0
    e = alpha * gt_n.dot(gt_v - Xt).view(1, 1)**2   + (1-alpha)*torch.linalg.norm(Xt - gt_v) 
    re = e
    # or using tukey like in DynFu paper.
    # e = gt_n.dot(Xt - gt_v).view(1,1)
    # re = robust_Tukey_penalty(e, torch.tensor(0.01))
    return re

def test_energy_2d():
    _ = torch.manual_seed(0)

    ## ============== INPUT ==============
    ### Constants
    knn = 3
    _radius = torch.tensor(0.025)
    ### Our inputs
    Xc = torch.tensor([[0,i,0] for i in range(1,6)]).flip(dims=(0,)).float()
    # Xc[:,0]=4.5
    Xc = Xc*_radius
    nc = torch.zeros_like(Xc)
    nc[:,0] = 1
    nc = nc.float()
    Xt1 = torch.tensor([[3,5,0], [4,4,0], [4.5,3,0], [4.5,2,0], [4.5,1,0]]).float()*_radius
    nt1 = cat_z0(produce_2d_normal_rightside(Xt1[:,:2]))
    Xt2 = torch.tensor([[3,5,0], [5,4,0], [4,3,0], [5,2,0], [3,1,0]]).float()*_radius
    nt2 = cat_z0(produce_2d_normal_rightside(Xt2[:,:2]))
    Xt = Xt1
    nt = nt1
    xc_gt = torch.stack([Xt, nt, Xc,nc], dim=1)

    ### Canonical 
    dgv = Xc 
    dgse, dgw = init_dgse_dgw(dgv, _radius)
    _kdtree = KDTree(dgv)
    target = Xt
    Tlw = torch.eye(4)
    node_to_nn = make_node_nn(Xc, _kdtree, knn)
    assert node_to_nn.shape == (5,knn)

    ## fields
    xv, yv = torch.meshgrid(
    torch.arange(0, 10),
    torch.arange(0, 10),
    )
    vox_coords = torch.stack([xv.flatten(), yv.flatten(), torch.zeros_like(xv.flatten())], dim=1).float()
    vox_coords = vox_coords*_radius
    volume_node_to_nn =  make_node_nn(vox_coords, _kdtree, knn)

    ## ============== PROCESS ==============
    # use vmap to avoid linear combination in batch dimension 
    warp_for_me_please = vmap(warp_helper, in_dims=(0, None, None, None, None, 0))
    energy_jac = jacrev(energy, argnums=3, has_aux=True)
    dqnorm_vmap = vmap(dqnorm, in_dims=0)
    se3_vmap = vmap(SE3, in_dims=(0))
    dq_vmap = vmap(SE3_dq, in_dims=(0))
    make_dq_from_vec_vmap = vmap(make_dq_from_vec, in_dims=(0))

    print("Start optimize! ")
    I = torch.eye(6).type_as(Tlw)  # to counter not full rank
    bs = 1  # res.shape[0]
    T_conj = torch.eye(4).float()
    T_conj[3,3] = - 1.0 

    # let's plot before learning
    plot_xc_xt_nt(Xc,Xt, nt,vox_coords, vis_dir=f"tests/vis_learning", step=999)
    for i in range(100):
        jse3, fx = energy_jac(xc_gt, Tlw, dgv, dgse, dgw, node_to_nn, node_to_nn)
        lmda = torch.mean(jse3.abs(), dim = 1)
        # lmda= 1e-2
        # print("done se3")
        j = jse3.view(bs, len(dgv), 1, 6)  # [bs,n_node,1,6]
        jT = custom_transpose_batch(j, isknn=True)  # [bs,n_node, 6,1]
        tmp_A = torch.einsum("bnij,bnjl->bnil", jT, j).view(
            bs * len(dgv), 6, 6
        )  # [bs*n_node,6,6]
        plot_heatmap_step(tmp_A.view(-1,6,6),vis_dir=f"tests/vis_heatmap",index=0,step=i)
        A = (tmp_A + torch.einsum('b,ij -> bij',lmda, I.view( 6, 6))).view(
            bs * len(dgv), 6, 6
        )  #  [bs*n_node,6,6]
        b = torch.einsum("bnij,bj->bni", jT, fx.view(bs, 1)).view(
            bs * len(dgv), 6, 1
        )  # [bs*n_node, 6, 1]
        # solved_delta = torch.linalg.lstsq(A, b)
        # A = torch.linalg.cholesky(A)
        # solved_delta = torch.cholesky_solve(b, A)
        # print(solved_delta.shape)
        solved_delta = torch.linalg.solve(A, b)
        solved_delta = post_process_solved_delta(solved_delta, bs, dgv, dgse)
        # update
        # dgse_tmp = se3_vmap(dgse)
        # sde_tmp = T_conj @ se3_vmap(solved_delta) @ T_conj 
        # print(solved_delta[4],'\n', sde_tmp[4])
        # out_tmp = torch.einsum("btij,btjk->bik", dgse_tmp, sde_tmp)
        # dgse = dq_vmap(out_tmp)
        dgse -= solved_delta
        # dgse -= 0.05*jse3
        # dgse = dqnorm_vmap(dgse.view(-1, 6)).view(len(dgv), 6)
        plot_heatmap_step(dgse.view(1,-1,6),vis_dir=f"tests/vis_dgse",index=0,step=i)
            
        dgse_dq = make_dq_from_vec_vmap(dgse)
        Xc_warp = warp_for_me_please(Xc, Tlw, dgv, dgse_dq, dgw, node_to_nn)
        vox_coords_live = warp_for_me_please(vox_coords, Tlw, dgv, dgse_dq, dgw, volume_node_to_nn)
        plot_xc_xt_nt(Xc_warp,Xt, nt,vox_coords_live, vis_dir=f"tests/vis_learning", step=i)
        print(
            "log: ",
            torch.sum(fx),
            lmda,
            # torch.min(jse3[jse3.abs() > 0].abs()),
            torch.mean(jse3),
            torch.mean(jse3.abs()),
        )


    ## ============== ASSERT the output ==============
    dgse_dq = make_dq_from_vec_vmap(dgse)
    for jj in range(5):
        dgv_nn = dgv[node_to_nn[jj]]
        dgw_nn = dgw[node_to_nn[jj]]
        dgse_nn = dgse_dq[node_to_nn[jj]]
        assert dgse_nn.shape == (knn,8)
        assert dgv_nn.shape == (knn,3)
        T = get_W(Xc[jj], Tlw, dgse_nn, dgw_nn, dgv_nn)
        R, t= decompose_se3(T)
        Xt =  (torch.einsum('bij,j->bi', R, Xc[jj]) + t.squeeze(-1)).squeeze()
        print(Xt)
        print(R)
        print(t)
        assert torch.allclose(Xt, target[jj], rtol=1e-1, atol=1e-02), f"{Xt} is different from {target[jj]}" # should pass

