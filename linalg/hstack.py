import numpy as np
import scipy.sparse as sps

def hstack(L1):
    L2 = []
    for elem in L1:
        if 1 == elem.ndim:
            L2.append(elem[:,np.newaxis])
        else:
            assert(2 == elem.ndim)
            L2.append(elem)

    return np.hstack(L2)
            
def sphstack(L1):
    L2 = []
    for elem in L1:
        if isinstance(elem,np.ndarray) and 1 == elem.ndim:
            L2.append(sps.csr_matrix(elem[:,np.newaxis]))
        else:
            assert(isinstance(elem, sps.spmatrix))
            L2.append(elem)

    return sps.hstack(L2)
