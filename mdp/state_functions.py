import numpy as np
from utils.parsers import KwargParser

class StateSpaceFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class InterpolatedFunction(StateSpaceFunction):
    """
    Take in a finite vector and a discretizer;
    Enable evaluation at arbitrary points via interpolation
    """
    def __init__(self,discretizer,y):
        N = discretizer.num_nodes
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

class ConstFn(StateSpaceFunction):
    def __init__(self,v):
        self.v = v
    def evaluate(self,points):
        (N,D) = points.shape 
        return self.v * np.ones(N)

class FixedVectorFn(StateSpaceFunction):
    def __init__(self,x):
        assert(1 == len(x.shape))
        self.x = x
    def evaluate(self,points):
        (N,D) = points.shape 

        assert((N,) == self.x.shape)
        return self.x    

class GaussianFn(StateSpaceFunction):
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
        
class BallSetFn(StateSpaceFunction):

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
        
class TargetZoneFn(StateSpaceFunction):

    def __init__(self,targets):
        assert((D,2) == targets.shape)
        self.targets = targets
        self.D = D
    
    def evaluate(self,points):
        (N,D) = points.shape
        assert(self.D == D)

        l_oob = points < self.target[:,0]
        r_obb = points > self.target[:,1]
        assert((N,) == l_oob.shape)
        assert((N,) == l_oob.shape)

        costs = np.zeros(N)
        costs[l_oob] = 1.0
        costs[r_oob] = 1.0
        
        return costs
