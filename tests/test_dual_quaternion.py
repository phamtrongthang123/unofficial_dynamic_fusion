import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch 

def check_unit_dq(dq):
    return torch.allclose(torch.linalg.norm(dq[:4]), torch.tensor(1.0), atol=1e-5) and torch.allclose(dq[:4].dot(dq[4:]), torch.tensor(0.0), atol=1e-6)
def test_normalization():
    from tmp_utils import dqnorm
    q = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    q_ = torch.tensor([1.5,3,4.5,6,4,2,0,-2]) /(15* torch.tensor(0.3).sqrt())
    re: torch.tensor = dqnorm(q)
    assert torch.allclose(re, q_, atol=1e-6)
    assert check_unit_dq(re)

def test_blending():
    v1 = torch.tensor([0.52,0.53,0.5])
    v2 = torch.tensor([0.45, 0.53,0.53])
    v3 = torch.tensor([0.5,0.51,0.51])
    v4 = torch.tensor([0.57,0.51,0.57])
    xc = torch.tensor([0.5,0.5,0.5])
    d=0.025 
    dgse1 = torch.tensor([0.1,0.3,0.5,0.16,0.23,0.27,0.22,0.27])
    dgse2 = torch.tensor([0.2,0.17,0.14,0,0.3,0.12,0.11,0.18])
    dgse3 = torch.tensor([0.22,0.2,0.1,0.3,0.23,0.2,0.6,0.38])
    dgse4 = torch.tensor([0.05,0.19,0.05,0.01,0.17,0.06,0.1,0.25])
    dgw = 1*d
    def blending_manual(v1, v2, v3,v4, xc,dgw, dgse1,dgse2, dgse3,dgse4):
        def w(v,xc,dgw):
            return torch.exp(-torch.linalg.norm(v-xc)**2 / (2*dgw**2))
        w1 = w(v1,xc,dgw)
        w2 = w(v2,xc,dgw)
        w3 = w(v3,xc,dgw)
        w4 = w(v4,xc,dgw)
        dgse = torch.zeros_like(dgse1)
        dgse += dgse1 * w1 
        dgse += dgse2 * w2 
        dgse += dgse3 * w3 
        dgse += dgse4 * w4 
        def dqnorm(dq):
            re = dq[:4]
            du = dq[4:]
            du = du/ torch.linalg.norm(re) - re / torch.linalg.norm(re) * re.dot(du)  / torch.linalg.norm(re)**2
            re = re/ torch.linalg.norm(re)
            return torch.cat((re,du))
        dgse = dqnorm(dgse)
        return dgse 
    a = blending_manual(v1, v2, v3,v4, xc,dgw, dgse1,dgse2, dgse3,dgse4)
    from tmp_utils import blending 
    dgv = torch.stack((v1,v2,v3,v4)).view(4,3)
    dgw = torch.tensor(dgw).view(1,1).repeat_interleave(4,0).view(4)
    dqs = torch.stack((dgse1,dgse2,dgse3,dgse4)).view(4,8)
    xc = xc.view(1,3)
    o = blending(xc,dqs,dgw,dgv)
    assert torch.allclose(a,o)
