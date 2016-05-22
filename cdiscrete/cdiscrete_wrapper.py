import cDiscrete as cd
import numpy as np

################################
# Check that the inputs are of the appropriate type

def check_vec(A):
    if not isinstance(A,np.ndarray):
        print '[!] Object not a np.ndarray'

    if not np.double == A.dtype:
        print '[!] Only supporting np.double exports' 
    
    return isinstance(A,np.ndarray) \
        and (np.double == A.dtype) \
        and (1 == len(A.shape))

def check_uvec(A):
    if not isinstance(A,np.ndarray):
        print '[!] Object not a np.ndarray'

    if not np.uint64 == A.dtype:
        print '[!] Only supporting np.double exports' 
    
    return isinstance(A,np.ndarray) \
        and (np.uint64 == A.dtype) \
        and (1 == len(A.shape))

def check_mat(A):
    if not isinstance(A,np.ndarray):
        print '[!] Object not a np.ndarray'

    if not np.double == A.dtype:
        print '[!] Only supporting np.double exports'
        
    return isinstance(A,np.ndarray) \
        and (np.double == A.dtype) \
        and (2 == len(A.shape))

def interpolate(vals,
                points,
                low,
                high,
                num_cells):
    # Type checking
    assert(check_vec(vals))
    assert(check_mat(points))
    assert(check_vec(low))
    assert(check_vec(high))
    assert(check_uvec(num_cells))
    
    # Size checking
    G = np.prod(num_cells + 1) + 1 # Num points + oob point
    assert(G == vals.size)

    # Everything else has to have D columns
    (N,D) = points.shape
    assert(D == low.size)
    assert(D == high.size)
    assert(D == num_cells.size)

    return cd.interpolate(vals,
                          points,
                          low,
                          high,
                          num_cells)

def argmax_interpolate(vals,
                       points,
                       low,
                       high,
                       num_cells):
    # Type checking
    assert(check_mat(vals))
    assert(check_mat(points))
    assert(check_vec(low))
    assert(check_vec(high))
    assert(check_uvec(num_cells))
    
    # Size checking
    G = np.prod(num_cells + 1) + 1 # Num points + oob point
    assert(G == vals.shape[0])

    # Everything else has to have D columns
    (N,D) = points.shape
    assert(D == low.size)
    assert(D == high.size)
    assert(D == num_cells.size)

    return cd.argmax_interpolate(vals,
                                 points,
                                 low,
                                 high,
                                 num_cells)

class SimulationOutcome:
    def __init__(self,points,
                 actions,
                 costs):
        self.points = points
        self.action = actions
        self.costs = costs

def simulate():
    (X,A,C) = cd.simulate_test()
    return SimulationOutcome(X,A,C)

def c_arange(N):
    return cd.c_arange(N)
