import torch 
import time 
from scipy.spatial import KDTree


t1 = time.time()
a = torch.randn(100,3)
kdtree = KDTree(a)
ann = [] 
for i in range(a.shape[0]):
    dists, idx = kdtree.query(a[i].cpu(), k=4, workers=-1)
    ann.append(torch.tensor(idx))
ann = torch.stack(ann).long()
print("run ", time.time() - t1)
t1 = time.time()
# a = torch.randn(100,3)
kdtree = KDTree(a)
ann_ = [] 
dists, idx = kdtree.query(a.cpu(), k=4, workers=-1)
ann_ = torch.tensor(idx).long()
print("run ", time.time() - t1)
assert ann_.shape == ann.shape, f'{ann_.shape} {ann.shape}'
assert torch.allclose(ann_,  ann)
print(ann_, ann)