import numpy as np
import scipy.sparse as sps

def csr_matrix_to_pickle_array(matrix):
    """
    Converts a CSR matrix into single np array with header
    """
    
    assert(type(matrix) == sps.csr_matrix)
    
    H = 6
    Flags = 0 # Not using this yet
    R = matrix.shape[0]
    C = matrix.shape[1]
    Dat = matrix.data.size
    Idx = matrix.indices.size
    Ipt = matrix.indptr.size

    N = sum([H,Dat,Idx,Ipt])
    A = np.empty(N)

    Writes = []
    Writes.append(np.array([Flags,R,C,Dat,Idx,Ipt]))
    Writes.append(matrix.data)
    Writes.append(matrix.indices)
    Writes.append(matrix.indptr)
        
    I = 0
    for write in Writes:
        J = I + write.size
        A[I:J] = write
        I = J
    assert(I == N) # Everything should be consumed
    
    return A
    
def pickle_array_to_csr_matrix(A):
    """
    Extracts CSR matrix from single np array with header
    """
    assert(1 == len(A.shape))
    N = A.size
    H = 6
    assert(N >= H)
    
    [Flags,R,C,Dat,Idx,Ipt] = A[0:H] 
    Lens = [Dat,Idx,Ipt]
    assert(N == (H + sum(Lens)))
    
    DII = [] # data,indices,indptr
    I = H
    for i in xrange(3):
        J = I + Lens[i]
        if i > 0:
            DII.append(A[I:J].astype(np.int))
        else:
            DII.append(A[I:J])
        I = J
    assert(I == N) # Everything should be consumed
    
    return sps.csr_matrix(tuple(DII),shape=(R,C))
    
def multi_matrix_to_pickle_array(Matrices):
    N = len(Matrices)    
    Pickles = []
    for M in Matrices:
        assert (2 == len(M.shape))
        Pickles.append(csr_matrix_to_pickle_array(M.tocsr()))
    PickleLens = [x.size for x in Pickles]
    A = np.empty(1 + N + sum(PickleLens))
    A[0] = N
    A[1:(N+1)] = PickleLens
    A[(N+1):] = np.concatenate(Pickles)
    return A
    
def pickle_array_to_multi_matrix(A):
    assert(abs(int(A[0]) - A[0]) < 1e-12)
    N = int(A[0])
    (Frac,Int) = np.modf(A[1:(N+1)])
    assert(np.linalg.norm(Frac) < 1e-9)
    Lens = Int.astype(np.int)
    
    Matrices = []
    I = N+1
    for i in xrange(N):
        J = I + Lens[i]
        Matrices.append(pickle_array_to_csr_matrix(A[I:J]))
        I = J
    assert(I == A.size) # Everything should be consumed
    return Matrices
    