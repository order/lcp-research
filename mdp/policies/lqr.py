class LinearFeedbackPolicy(Policy):
    """
    Linear feedback gain policy; think LQR
    """
    def __init__(self,gains,offset,**kwargs):
        (n,m) = gains.shape
        assert((m,) == offset.shape)
        
        self.K = gains
        self.offset = offset 
        self.lim = kwargs.get('limit',(-float('inf'),float('inf')))
        
    def get_decisions(self,states):
        m = len(states.shape) # Remap vectors to row vectors
        if 1 == m:
            states = states[np.newaxis,:] 
        
        (NumPoints,NumDim) = states.shape
        (NumActions,D) = self.K.shape
        assert(NumDim == D)
        
        decisions = -self.K.dot(states.T - self.offset[:,np.newaxis]).T # u = -Kx
        decisions = np.maximum(self.lim[0],np.minimum(self.lim[1],decisions))
        assert((NumPoints,NumActions) == decisions.shape)
        
        return decisions 
