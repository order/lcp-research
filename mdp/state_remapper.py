import numpy as np
import math

class StateRemapper(object):
    """
    Abstract class defining custom state-to-state remapping. 
    For example, if the velocity of an object is capped in [-B,B], then
    the state remapper might set the velocity component of the state to min(B,max(v,-B)).
    (See RangeThreshStateRemapper)
    
    Also used as the base class for any dynamics
    """
    def remap(self,states):
        """
        All state remappers must implement this.
        """
        raise NotImplementedError()
        
class RangeThreshStateRemapper(StateRemapper):
    """
    Thresholds a components of a state to be within a range
    """    
    def __init__(self,dim,low,high):

        self.dim = dim
        self.low = low
        self.high = high
        self.eps = 1e-6
        
    def remap(self, states):
        """
        Projects entries onto the given range
        
        Assumes states to be an N x d np.array
        """
        states[states[:,self.dim] > self.high,self.dim] = self.high - self.eps
        states[states[:,self.dim] < self.low,self.dim] = self.low + self.eps
        return states
        
    def __str__(self):
        return 'RangeThreshStateRemapper (dim={0},low={1},high={2})'.format(self.dim,self.low,self.high)
        
class AngleWrapStateRemaper(StateRemapper):
    def __init__(self,dim):
        self.dim = dim
        self.eps = 1e-9
        
    def remap(self,states):
        states[:,self.dim] = np.mod(states[:,self.dim], 2.0*math.pi - 1e-9)
        assert(not np.any(states[:,self.dim] > 2.0*math.pi))
        assert(not np.any(states[:,self.dim] < 0.0))
        return states
        
    def __str__(self):
        return 'AngleWrapStateRemaper (dim={0})'.format(self.dim)
        
class WrapperStateRemaper(StateRemapper):
    def __init__(self,dim,low,high):
        self.dim = dim
        self.eps = 1e-9
        self.low = low
        self.high = high
        
    def remap(self,states):
        mask = np.logical_or(states[:,self.dim] > self.high,states[:,self.dim] < self.low)
        states[mask,self.dim] = np.mod(states[mask,self.dim] - self.low, self.high - self.low) + self.low
        assert(not np.any(states[mask,self.dim] > self.high))
        assert(not np.any(states[mask,self.dim] < self.low))
        return states
        
    def __str__(self):
        return 'AngleWrapStateRemaper (dim={0})'.format(self.dim)
