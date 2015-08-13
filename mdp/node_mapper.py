def NodeMapper(object):
    """
    Abstract class defining custom state-to-node mapping. 
    For example, OOB states.
    """
    def map(self,states):
        """
        All node mappers must implement this.
        """
        raise NotImplementedError()
        
class OOBSinkNodeMapper(StateRemapper):
    def __init__(self,dim,low,high,sink_node):
        """
        Ensures that a state doesn't excede an (axis-aligned) boundary by sending to sink state
        """
        self.dim = dim
        self.low = low
        self.high = high
        self.sink_node = sink_node
        
    def remap(self, states):
        """
        Assumes states to be an N x d np.array
        """
        Mapping = {}
        for (i,state_comp) in in enumerate(X[:,self.dim]):
            if self.low <= state_comp <= self.high:
                continue
            Mapping[i] = self.sink_node
        return Mapping
        