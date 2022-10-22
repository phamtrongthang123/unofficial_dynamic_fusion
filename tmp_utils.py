import numpy as np 
from numpy import linalg as la
from scipy.spatial import KDTree
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
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

# Construct deformation graph from canonical vertices
def construct_graph(self):
    # uniform sampling
    nodes_v, nodes_idx = uniform_sample(self._vertices, self._radius)


    '''
    Each node is a 4-tuple (index of corresponding surface vertex dg_idx, 3D position dg_v, 4x4 Transformation dg_se3, weight dg_w) 
    HackHack:
    Not sure how to determine dgw. Use sample radius for now.
    '''
    for i in range(len(nodes_v)):
        self._nodes.append((nodes_idx[i],
                            nodes_v[i],
                            np.array([1, 0.00, 0.00, 0.00, 0.00, 0.01, 0.01, 0.00], dtype=np.float32),
                            2 * self._radius))

    # construct kd tree
    self._kdtree = KDTree(nodes_v)
    self._neighbor_look_up = []
    for vert in self._vertices:
        dists, idx = self._kdtree.query(vert, k=self._knn)
        self._neighbor_look_up.append(idx)