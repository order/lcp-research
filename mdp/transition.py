import numpy as np

class TransitionFunction(object):
    """
    Abstract class defining (state,action)-to-state mapping. 

    Transition functions may be stochastic; setting number of samples
    should yield an independent redraw
    """
    def transition(self,states,action,samples=1):
        """
        All state remappers must implement this.
        """
        raise NotImplementedError()
        
    def single_transition(self,point,action):
        assert(1 == len(point.shape))
        (N,) = point.shape
        states = self.transition(point[np.newaxis,:],
                                 action,1)
        assert((1,1,N) == states.shape)
        return states[0,0,:]
        
