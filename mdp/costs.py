import numpy as np

class QuadraticCost(object):
    """
    Quadratic cost for being away from the origin.
    Has an override cost for any states with NaN
    """
    def __init__(self,coeff,setpoint,**kwargs):
        self.coeff = coeff
        self.override = kwargs.get('override',None)

        assert(1 == len(setpoint.shape))
        self.setpoint = np.reshape(setpoint,(1,setpoint.size))
    
    def cost(self,points,action):
        
        costs = np.power(points - self.setpoint, 2).dot(self.coeff)
        
        # Deal with overrides
        mask = np.isnan(costs)
        if not self.override:
            assert(not np.any(mask))
        else:
            costs[mask] = self.override
        assert(not np.any(np.isnan(costs)))
        
        return costs
        
class BallCost(object):

    def __init__(self,center,radius,**kwargs):
        self.center = center
        self.radius = radius
        self.in_cost = kwargs.get('in_cost',0.0)
        self.out_cost = kwargs.get('out_cost',1.0)

        assert(1 == len(center.shape))
    
    def cost(self,points,action):
        (N,d) = points.shape
        inbound = (np.linalg.norm(points - self.center,axis=1) <= self.radius)
        assert((N,) == inbound.shape)
        
        costs = np.empty(points.shape[0])
        costs[inbound] = self.in_cost
        costs[np.logical_not(inbound)] = self.out_cost
        
        return costs
        
class TargetZoneCost(object):

    def __init__(self,targets,**kwargs):
        self.targets = targets
        self.in_cost = kwargs.get('in_cost',0.0)
        self.out_cost = kwargs.get('out_cost',1.0)
    
    def cost(self,points,action):
        (N,D) = points.shape
        assert((2,D) == self.targets.shape)
        
        costs = self.out_cost*np.ones(N)
        
        goal_mask = np.ones(N,dtype='bool')
        for d in xrange(D):
            goal_mask = np.logical_and(goal_mask,points[:,d] >= self.targets[0,d])
            goal_mask = np.logical_and(goal_mask,points[:,d] <= self.targets[1,d])
        assert((N,) == goal_mask.shape)
        costs[goal_mask] = self.in_cost
        
        return costs