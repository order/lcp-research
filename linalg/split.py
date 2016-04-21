import numpy as np

def split(A,max_size,axis=0):
    N = A.shape[0]
    num_splits = np.ceil(float(N) / float(max_size))
    
    return np.array_split(A,num_splits,axis)
