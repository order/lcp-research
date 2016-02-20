import numpy as np

def trace(A):
    if type(A) == np.ndarray:
        return np.trace(A)
    (I,J) = A.nonzero()

    N = I.size

    acc = 0.0
    for k in xrange(N):
        if I[k] == J[k]:
            acc += A[I[k],J[k]]
    return acc
