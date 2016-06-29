import numpy as np
import matplotlib.pyplot as plt
def orthonorm(B):
    #Ortho
    [Q,R] = np.linalg.qr(B)
    #indep = (np.abs(np.diag(R)) > 1e-15)
    return Q
