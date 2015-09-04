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
        
    def remap(self, states):
        """
        Projects entries onto the given range
        
        Assumes states to be an N x d np.array
        """
        states[states[:,self.dim] > self.high] = self.high
        states[states[:,self.dim] < self.low] = self.low
        return states
        
class AngleWrapStateRemaper(StateRemapper):
    def __init__(self,dim):
        self.dim = dim
        
    def remap(self,states):
        states[:,self.dim] = np.mod(states[:,self.dim], 2.0*math.pi)
        return states