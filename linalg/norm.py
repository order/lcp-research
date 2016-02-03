import scipy.sparse as sps
import scipy as sp
import numpy as np

def norm(X,norm='F'):
    assert(norm.lower() in ['f','fro'])
    
    return np.abs(X.sum())
    
    
