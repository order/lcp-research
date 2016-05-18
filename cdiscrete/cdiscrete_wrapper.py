import cDiscrete as cd
import numpy as np

################################
# Check that the inputs are of the appropriate type

def export_check_vec(A):
    if not isinstance(A,np.ndarray):
        print '[!] Object not a np.ndarray'

    if not np.double == A.dtype:
        print '[!] Only supporting np.double exports' 
    
    return isinstance(A,np.ndarray) \
        and (np.double == A.dtype) \
        and (1 == len(A.shape))

def export_check_mat(A):
    if not isinstance(A,np.ndarray):
        print '[!] Object not a np.ndarray'

    if not np.double == A.dtype:
        print '[!] Only supporting np.double exports'
        
    return isinstance(A,np.ndarray) \
        and (np.double == A.dtype) \
        and (2 == len(A.shape))       

def incr(A):
    assert(export_check_mat(A))
    return cd.incr(A)
        
