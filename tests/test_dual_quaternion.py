def test_normalization():
    q = torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    q_ = torch.tensor([1.5,3,4.5,6,4,2,0,-2]) /(15* torch.tensor(0.3).sqrt())
    re = dqnorm(q)
    assert torch.allclose(re, q_)