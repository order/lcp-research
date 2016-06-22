import numpy as np
import linalg

def saturate(x,l,u):
    T = np.maximum(x,l)
    T = np.minimum(T,u)
    return T

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

class DoubleIntBoundary(Boundary):
    def __init__(self,boundary):
        self.boundary = boundary
        assert(2 == len(boundary))
        for (l,u) in boundary:
            assert(l < u)
        self.lower_bound = [l for (l,u) in boundary]
        self.upper_bound = [u for (l,u) in boundary]

    def enforce(self,states):
        # Enforce the boundary by saturation
        assert(2 <= len(states.shape) <= 3)
        # 2 -> number of points x dimension
        # 3 -> samples x number of points x dimension
        
        assert(2 == states.shape[-1])

        X = states[...,0]
        V = states[...,1]
        (lox,lov) = self.lower_bound
        (hix,hiv) = self.upper_bound
        
        X = ((X - lox) % (hix - lox)) + lox
        V = saturate(V,lov,hiv)

        R=np.empty(states.shape)
        R[...,0] = X
        R[...,1] = V
        
        return R

    def random_points(self,N):
        return linalg.random_points(self.boundary,N)


class SelectSaturationBoundary(SaturationBoundary):
    def __init__(self,boundary,dims):
        self.boundary = boundary
        self.D = len(boundary)
        for (l,u) in boundary:
            assert(l < u)
        self.lower_bound = [l for (l,u) in boundary]
        self.upper_bound = [u for (l,u) in boundary]
        self.dims = dims
        assert(len(dims) <= self.D)

    def enforce(self,states):
        # Enforce the boundary by saturation
        assert(2 <= len(states.shape) <= 3)
        # 2 -> number of points x dimension
        # 3 -> samples x number of points x dimension

        old_shape = states.shape
        D = states.shape[-1]
        assert(D == self.D)
        for d in self.dims:
            states[...,d] = np.maximum(states[...,d],self.lower_bound[d])
            states[...,d] = np.minimum(states[...,d],self.upper_bound[d])
        assert(old_shape == states.shape)
        return states

class HillcarBoundary(Boundary):
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

        old_shape = states.shape
        
        vec = False
        if 1 == (len(states.shape)):
            vec = True
            states = states[np.newaxis,:]

        D = states.shape[-1]
        assert(2 == D)
        
        T = np.maximum(states,self.lower_bound)
        assert(states.shape == T.shape)
        T = np.minimum(T,self.upper_bound)

        mask = (np.abs(T[...,0] - states[...,0]) > 1e-12)
        T[mask,1] = 0 # boundary
        
        if vec:
            T = T[0,:]
        assert(T.shape == states.shape)
        return T
        

    def random_points(self,N):
        return linalg.random_points(self.boundary,N)

# WrapBoundary -> torus
# CompoundBoundary -> use list of state remappers to implement
