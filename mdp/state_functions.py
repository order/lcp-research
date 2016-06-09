import numpy as np
from utils.parsers import KwargParser

class RealFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class MultiFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class InterpolatedFunction(RealFunction):
    """
    Take in a finite vector and a discretizer;
    Enable evaluation at arbitrary points via interpolation
    """
    def __init__(self,discretizer,y):
        N = discretizer.num_nodes()
        assert((N,) == y.shape)
        self.target_values = y
        self.discretizer = discretizer
        
    def evaluate(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]

        y = self.target_values
        (N,) = y.shape
        (M,d) = states.shape
        
        # Convert state into node dist
        P = self.discretizer.points_to_index_distributions(states)
        assert((N,M) == P.shape)
        f = (P.T).dot(y)
        assert((M,) == f.shape)
        return f

class InterpolatedMultiFunction(MultiFunction):
    """
    Take in a finite vector and a discretizer;
    Enable evaluation at arbitrary points via interpolation
    """
    def __init__(self,discretizer,y):
        self.target_values = y
        self.discretizer = discretizer
        
    def evaluate(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]

        y = self.target_values
        (N,A) = y.shape
        (M,d) = states.shape
        
        # Convert state into node dist
        P = self.discretizer.points_to_index_distributions(states)
        assert((N,M) == P.shape)
        f = (P.T).dot(y)
        assert((M,A) == f.shape)
        return f

class ConstFn(RealFunction):
    def __init__(self,v):
        self.v = v
    def evaluate(self,points):
        (N,D) = points.shape 
        return self.v * np.ones(N)

class FixedVectorFn(RealFunction):
    def __init__(self,x):
        assert(1 == len(x.shape))
        self.x = x
    def evaluate(self,points):
        (N,D) = points.shape 

        assert((N,) == self.x.shape)
        return self.x    

class GaussianFn(RealFunction):
    def __init__(self,bandwidth,
                 setpoint,
                 non_physical):
        
        assert(1 == len(setpoint.shape))
        N = setpoint.size
        self.setpoint = np.reshape(setpoint,(1,N))
        self.bandwidth = bandwidth
        self.non_phys = non_physical
 
    def evaluate(self,points):
        (N,D) = points.shape 
        
        norm = np.sum(np.power(points - self.setpoint,2),axis=1)
        costs = np.exp(-norm / self.bandwidth)
        assert((N,) == costs.shape)

        # Deal with overrides
        mask = np.isnan(costs)
        costs[mask] = self.non_phys
        
        return costs
        
class BallSetFn(RealFunction):

    def __init__(self,center,radius):
        
        self.center = center
        self.radius = radius
    
    def evaluate(self,points):       
        (N,d) = points.shape
        nan_mask = ~np.any(np.isnan(points),axis = 1)
        assert((N,) == nan_mask.shape)
        n = nan_mask.sum()

        inbound = np.zeros(N,dtype=bool)
        diff = points[nan_mask,:] - self.center
        assert((n,d) == diff.shape)
        r = np.linalg.norm(diff, axis=1)
        assert((n,) == r.shape)
        inbound[nan_mask] = (r <= self.radius)
        
        costs = np.empty(points.shape[0])
        costs[inbound] = 0.0
        costs[~inbound] = 1.0
        
        return costs
        
class TargetZoneFn(RealFunction):

    def __init__(self,targets):
        (D,_) = targets.shape
        self.targets = targets
        self.D = D
    
    def evaluate(self,points):
        (N,D) = points.shape
        assert(self.D == D)

        mask = np.any(np.isnan(points),axis=1)
        mask[~mask] = np.any(points[~mask,:] < self.targets[:,0],
                             axis=1)
        mask[~mask] = np.any(points[~mask,:] > self.targets[:,1],
                             axis=1)        

        costs = np.zeros(N)
        costs[mask] = 1.0

        return costs
