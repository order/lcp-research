import scipy.sparse as sps
import scipy as sp
import numpy as np

def norm(M):
    if isinstance(M,sps.coo_matrix):
        return coo_F_norm(M)
    else:
        raise NotImplementedError()

def coo_F_norm(M):
    assert(isinstance(M,sps.coo_matrix))
    return np.sum(np.abs(M.data))
