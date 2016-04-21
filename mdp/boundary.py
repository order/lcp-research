import numpy as np
import linalg
class Boundary(object):
    def enforce(self,states):
        raise NotImplmentedError()

class SaturationBoundary(Boundary):
    def __init__(self,boundary):
        self.boundary = boundary
        self.D = len(boundary)
        for (l,u) in boundary:
            assert(l < u)
        self.lower_bound = [l for (l,u) in boundary]
        self.upper_bound = [u for (l,u) in boundary]

    def enforce(self,states):
        # Enforce the boundary by saturation
        assert(2 <= len(states.shape) <= 3)
        # 2 -> number of points x dimension
        # 3 -> samples x number of points x dimension
        
        D = states.shape[-1]
        assert(D == self.D)

        T = np.maximum(states,self.lower_bound)
        assert(states.shape == T.shape)
        return np.minimum(T,self.upper_bound)

    def random_points(self,N):
        return linalg.random_points(self.boundary,N)
    
# WrapBoundary -> torus
# CompoundBoundary -> use list of state remappers to implement
