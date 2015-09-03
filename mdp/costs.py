import numpy as np

class QuadraticCost(object):
    """
    Quadratic cost for being away from the origin.
    Has an override cost for any states with NaN
    """
    def __init__(self,coeff,**kwargs):
        self.coeff = coeff
        self.override = kwargs.get('override',None)
        self.setpoint = kwargs.get('setpoint',None)
        if 1 == len(self.setpoint.shape):
            # Ensure a row vector
            self.setpoint = np.reshape(self.setpoint,(1,self.setpoint.size))
    
    def cost(self,points,action):
        # Init setpoint to origin if not set
        if not self.setpoint:
            (N,d) = points.shape
            self.setpoint = np.zeros((1,d))
            
        costs = np.pow(points - self.setpoint, 2).dot(self.coeff)
        
        # Deal with overrides
        mask = np.isnan(costs)
        if not self.override:
            assert(not np.any(mask))
        else:
            costs[mask] = self.override
        assert(not np.any(np.isnan(costs)))
        
        return costs