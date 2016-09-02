import numpy as np
import scipy.sparse as sps

from lcp import LCPObj

def read_sp_mat(sp_mat_file):
    data = np.fromfile(sp_mat_file)
    R = int(data[0])
    C = int(data[1])
    NNZ = int(data[2])
    assert (3 + 3*NNZ,) == data.shape

    data = np.reshape(data[3:],(NNZ,3))
    
    S = sps.coo_matrix((data[:,2],(data[:,0], data[:,1])),shape=(R,C))
    return S

def read_mat(mat_file):
    data = np.fromfile(mat_file)
    R = int(data[0])
    C = int(data[1])
    assert data.size == (R*C+2)

    return np.reshape(data[2:],(R,C))
def read_vec(vec_file):
    return np.fromfile(vec_file);

def build_mdp_lcp():
    P1 = read_sp_mat("mesh/p_neg.spmat")
    P2 = read_sp_mat("mesh/p_pos.spmat")
    C = read_mat("mesh/costs.mat")
    W = read_vec("mesh/weights.vec")

    (N,AD) = C.shape
    assert 2 == AD
    assert (N,) == W.shape
    assert (N,N) == P1.shape
    assert (N,N) == P2.shape

    I = sps.eye(N)
    g = 0.997
    M = sps.bmat([[None,I - g*P1,I - g*P2],
                  [g*P1.T - I,None,None],
                  [g*P1.T - I,None,None]])
    q = np.hstack([-W,C[:,0],C[:,1]])

    return LCPObj(M,q)
    
if __name__ == "__main__":
    lcp = build_mdp_lcp()

    print lcp
    
