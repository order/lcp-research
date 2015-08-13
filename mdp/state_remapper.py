class StateRemapper(object):
    """
    Abstract class defining custom state-to-state remapping. 
    For example, if the velocity of an object is capped in [-B,B], then
    the state remapper might set the velocity component of the state to min(B,max(v,-B)).
    (See RangeThreshStateRemapper)
    """
    def remap(self,states):
        """
        All state remappers must implement this.
        """
        raise NotImplementedError()
        
class RangeThreshStateRemapper(StateRemapper):
    def __init__(self,dim,low,high):
        """
        Thresholds a components of a state to be within a range
        """
        self.dim = dim
        self.low = low
        self.high = high
        
    def remap(self, states):
        """
        Assumes states to be an N x d np.array
        """
        states[states[:,d] > self.high] = self.high
        states[states[:,d] < self.low] = self.low
        return states