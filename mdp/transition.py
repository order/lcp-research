import numpy as np

class TransitionFunction(object):
    """
    Abstract class defining (state,action)-to-state mapping. 

    Transition functions may be stochastic; setting number of samples
    should yield an independent redraw
    """
    def multisample_transition(self,states,action,samples):
        """
        Most general function; transitions multiple samples
        for multiple states with multiple actions

        Individual transition functions should implement this
        """
        raise NotImplementedError()
    
    def transition(self,states,action):
        """
        Default single sample transition based on the 
        multisample version
        """
        return self.multisample_transition(states,
                                           action,
                                           1)[0,:,:]
    
        
    def single_transition(self,point,action):
        assert(1 == len(point.shape))
        (N,) = point.shape
        state = self.transition(point[np.newaxis,:], action)
        assert((1,N) == state.shape)
        return state[0,:]
        
