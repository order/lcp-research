import numpy as np
from utils.parsers import KwargParser

class StateSpaceFunction(object):
    def eval(self,points,**kwargs):
        raise NotImplementedError()

class ConstFn(StateSpaceFunction):
    def __init__(self,v):
        self.v = v
    def eval(self,points,**kwargs):
        (N,D) = points.shape 

        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)

        return self.v * np.ones(N)

class FixedVectorFn(StateSpaceFunction):
    def __init__(self,x):
        assert(1 == len(x.shape))
        self.x = x
    def eval(self,points,**kwargs):
        (N,D) = points.shape 

        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)

        assert((N,) == self.x.shape)
        return self.x
    

class GaussianBowlFn(StateSpaceFunction):
    """
    Has a bowl with low-point 0
    """

    def __init__(self,bandwidth,setpoint,**kwargs):
        parser = KwargParser()
        parser.add('override',None) # Should be a float or int if set
        args = parser.parse(kwargs)
        self.override = args['override']  

        assert(1 == len(setpoint.shape))
        N = setpoint.size
        self.setpoint = np.reshape(setpoint,(1,N))
        self.bandwidth = bandwidth
 
    def eval(self,points,**kwargs):
        (N,D) = points.shape 

        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)
        
        # 1 - exp(-||x - m||^2 / b)
        norm = np.sum(np.power(points - self.setpoint,2),axis=1) # ||x-m||^2
        costs = 1 - np.exp(-norm / self.bandwidth)
        print costs.shape
        assert((N,) == costs.shape)

        # Deal with overrides
        mask = np.isnan(costs)
        if not self.override:
            assert(not np.any(mask))
        else:
            costs[mask] = self.override
        assert(not np.any(np.isnan(costs)))
        
        return costs

class QuadraticFn(StateSpaceFunction):
    """
    Quadratic cost for being away from the origin.
    Has an override cost for any states with NaN
    """
    def __init__(self,coeff,setpoint,**kwargs):
        parser = KwargParser()
        parser.add('override',None) # Should be a float or int if set
        args = parser.parse(kwargs)
        self.override = args['override']

        self.coeff = coeff
        assert(1 == len(coeff.shape))
        N = coeff.size        

        assert((N,) == setpoint.shape)
        self.setpoint = np.reshape(setpoint,(1,N))
    
    def eval(self,points,**kwargs):
        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)
        
        costs = np.power(points - self.setpoint, 2).dot(self.coeff)
        
        # Deal with overrides
        mask = np.isnan(costs)
        if not self.override:
            assert(not np.any(mask))
        else:
            costs[mask] = self.override
        assert(not np.any(np.isnan(costs)))
        
        return costs
        
class BallSetFn(StateSpaceFunction):

    def __init__(self,center,radius):
        
        self.center = center
        self.radius = radius
        self.in_cost = 0.0
        self.out_cost = 1.0

        assert(1 == len(center.shape))
    
    def eval(self,points,**kwargs):
        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)
        
        (N,d) = points.shape
        nan_mask = ~np.any(np.isnan(points),axis = 1)
        assert((N,) == nan_mask.shape)

        inbound = np.zeros(N,dtype=bool)
        r = np.linalg.norm(points[nan_mask,:] - self.center,axis=1)
        inbound[nan_mask] = (r <= self.radius)
        
        costs = np.empty(points.shape[0])
        costs[inbound] = self.in_cost
        costs[~inbound] = self.out_cost
        
        return costs
        
class TargetZoneFn(StateSpaceFunction):

    def __init__(self,targets):
        self.targets = targets
        self.in_cost = 0.0
        self.out_cost = 1.0
    
    def eval(self,points,**kwargs):
        parser = KwargParser()
        parser.add_optional('action') #Ignored
        args = parser.parse(kwargs)
        
        (N,D) = points.shape
        assert((2,D) == self.targets.shape)
        
        costs = self.out_cost*np.ones(N)
        
        goal_mask = np.ones(N,dtype='bool')
        for d in xrange(D):
            goal_mask = np.logical_and(goal_mask,
                                       points[:,d] >= self.targets[0,d])
            goal_mask = np.logical_and(goal_mask,
                                       points[:,d] <= self.targets[1,d])
        assert((N,) == goal_mask.shape)
        costs[goal_mask] = self.in_cost
        
        return costs
