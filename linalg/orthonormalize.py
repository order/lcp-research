import numpy as np
def orthonorm(B):
    #Ortho
    [Q,R] = np.linalg.qr(B)
    indep = (np.abs(np.diag(R)) > 1e-15)
    return Q[:,indep]
