import numpy as np

class QuadraticCost(object):
    """
    Quadratic cost for being away from the origin.
    Has an override cost for any states with NaN
    """
    def __init__(self,coeff,override):
        self.coeff = coeff
        self.override = override
    
    def cost(self,points,action):       
        costs = (points**2).dot(self.coeff)
        mask = np.isnan(costs)
        costs[mask] = self.override
        
        assert(not np.any(np.isnan(costs)))
        
        return costs