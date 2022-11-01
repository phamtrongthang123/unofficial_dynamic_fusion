import torch 
import time 
from scipy.spatial import KDTree
from tmp_utils import SE3, dqnorm, compose_se3
from functorch import vmap, jacrev



def quat_from_rot(_rot):
    tr = _rot[0,0] + _rot[1,1] + _rot[2,2]
    ifhere = torch.stack((tr>0, torch.logical_and(_rot[0,0] > _rot[1,1],_rot[0,0] > _rot[2,2]), _rot[1,1] > _rot[2,2], torch.tensor(True))).long().view(-1)
    print(ifhere)
    s = torch.sqrt(tr + 1.0) * 2
    tmp1 = []
    tmp1.append(s*0.25)
    tmp1.append((_rot[2,1] - _rot[1,2]) / s)
    tmp1.append((_rot[0,2] - _rot[2,0]) / s)
    tmp1.append((_rot[1,0] - _rot[0,1]) / s)
    re = torch.stack(tmp1).view(1,4)
    s = torch.sqrt(1.0 + _rot[0,0] - _rot[1,1] - _rot[2,2]) * 2
    w = (_rot[2,1] - _rot[1,2]) / s
    x = 0.25*s
    y = (_rot[0,1] + _rot[1,0]) / s
    z = (_rot[0,2] + _rot[2,0]) / s
    re = torch.cat((re,torch.stack((w,x,y,z)).view(1,4)), dim=0)
    s = torch.sqrt(1.0 + _rot[1,1] - _rot[0,0] - _rot[2,2]) * 2
    w = (_rot[0,2] - _rot[2,0]) / s
    x = (_rot[0,1] + _rot[1,0]) / s
    y = 0.25*s
    z = (_rot[1,2] + _rot[2,1]) / s
    re = torch.cat((re,torch.stack((w,x,y,z)).view(1,4)), dim=0)
    s = torch.sqrt(1.0 + _rot[2,2] - _rot[0,0] - _rot[1,1]) * 2
    w = (_rot[1,0] - _rot[0,1]) / s
    x = (_rot[0,2] + _rot[2,0]) / s
    y = (_rot[1,2] + _rot[2,1]) / s
    z = 0.25*s
    re = torch.cat((re,torch.stack((w,x,y,z)).view(1,4)), dim=0)
    print(torch.any(torch.isinf(re), dim = 1))
    re_withoutinf = torch.where(torch.any(torch.isinf(re), dim = 1).view(-1,1), torch.tensor([1,0,0,1]).type_as(_rot), re)
    re_withoutnan = torch.where(torch.any(torch.isnan(re_withoutinf), dim = 1).view(-1,1), torch.tensor([1,0,0,1]).type_as(_rot), re_withoutinf)
    print(re_withoutnan)
    ree = torch.matmul((torch.logical_and(ifhere, torch.cumsum(ifhere, dim = 0) == 1)).float(), re_withoutnan)
    print(re[3] * 0)
    print(ree)
    return ree

def quat_add(q0, q1):
    return q0 + q1 

def quat_mul(_q0, _q1):
    w = _q0[0]*_q1[0] - _q0[1]*_q1[1] - _q0[2]*_q1[2] - _q0[3]*_q1[3]
    x = _q0[0]*_q1[1] + _q0[1]*_q1[0] + _q0[2]*_q1[3] - _q0[3]*_q1[2]
    y = _q0[0]*_q1[2] - _q0[1]*_q1[3] + _q0[2]*_q1[0] + _q0[3]*_q1[1]
    z = _q0[0]*_q1[3] + _q0[1]*_q1[2] - _q0[2]*_q1[1] + _q0[3]*_q1[0]    
    return torch.stack((w,x,y,z))

def dual_quat_mul(q0,q1):
    quat0 = quat_mul(q0[:4],q1[:4])
    quat1 = quat_add(quat_mul(q0[:4], q1[4:]), quat_mul(q0[4:], q1[:4]))
    return torch.cat((quat0, quat1))

def SE3_dq(mat44):
    R = mat44[:3, :3]
    t = mat44[:3,3]
    rot_part = torch.cat((quat_from_rot(R), torch.zeros(4)))
    vec_part = torch.tensor([1,0,0,0,0,0.5*t[0], 0.5*t[1], 0.5*t[2]])
    return dual_quat_mul(vec_part, rot_part)

def quat_conj(quat):
    return torch.tensor([quat[0], -quat[1], -quat[2], -quat[3]])

def mat34(dq):
    dq_norm = dqnorm(dq)
    r = SE3(dq).view(4,4)[:3,:3]
    vec_part = 2.0*quat_mul(dq_norm[4:],quat_conj(dq_norm[:4]))
    t = vec_part[1:]
    print(t.shape, r.shape)
    
    return torch.cat((torch.cat((r,t.view(3,1)), dim=1), torch.tensor([0,0,0,1]).view(1,4)), dim = 0)

dqq = torch.randn(8)
# a = mat34(dqnorm(dqq)).view(4,4)
# dq = SE3_dq(a)
# _a = mat34(dq)
# dq = SE3_dq(_a)
# _a1 = mat34(dq)
# assert torch.allclose(_a1, _a), f'{_a1, _a}'
# for i in range(10):
#     print(_a)
#     print(dq)
#     dq = SE3_dq(_a)
#     _a = SE3(dq).view(4,4)


R = torch.randn(10,3,3)
vmapp = vmap(quat_from_rot, in_dims=0)
aa = vmapp(R)

# nodes_w = torch.tensor(2.0).view(1,1).repeat_interleave(a.shape[0], 0).view(-1)
# print(nodes_w.shape)
# print(torch.sqrt(torch.tensor(-1)))

# a = torch.randn(4,8)
# a[1] = a[1]/0 
# b = torch.where(torch.isinf(a), -11, torch.tensor([1,0,0,0,0,0,0,0]))
# print(b)


b = torch.tensor([[ 9.9971e-01, -6.6122e-05,  2.4009e-02, -3.4403e-02],
              [ 2.3128e-04,  9.9998e-01, -6.8816e-03,  3.5100e-03],
              [-2.4008e-02,  6.8852e-03,  9.9969e-01,  2.2866e-03],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
b = torch.tensor([[ 9.9981e-01, -3.1563e-04,  1.9341e-02, -2.8007e-02],
              [ 4.4879e-04,  9.9998e-01, -6.8874e-03,  3.4571e-03],
              [-1.9339e-02,  6.8948e-03,  9.9979e-01,  1.5284e-03],
              [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
print(quat_from_rot(b))


a = torch.tensor([[1,2,3], [0,0,0], [4,5,6]]).float()
print(torch.where((torch.mean(a, dim = 1) ==0).view(-1,1), a+ 10*torch.tensor([1.0,0.0,0.0]), a + torch.mean(a, dim=1).view(-1,1)), torch.mean(a, dim=(0,1)))
b = torch.randn(2,1,1)
print( b* torch.eye(8).view(1,8,8), b)