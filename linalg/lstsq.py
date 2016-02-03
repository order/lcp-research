import numpy as np
import scipy as sp
import scipy.sparse as sps

def lsmr_matrix(A,B,**kwargs):
    """
    A wrapper around sparse.scipy.linalg.lsmr that handles
    matrices AX = B (rather than just vectors Ax = b)
    """
    (n,k) = A.shape
    (N,M) = B.shape
    assert(N == n)
    
    X = np.empty((k,M)) # Allocate the X matrix (dense)
    
    for i in xrange(M):
        b = B[:,i].toarray()
        ret = sps.linalg.lsmr(A,b,**kwargs)
        assert((k,) == ret[0].shape)
        X[:,i] = ret[0]

    return X
