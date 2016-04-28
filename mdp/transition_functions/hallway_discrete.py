import numpy as np
import scipy as sp
import scipy.sparse as sps

from mdp.transition import TransitionFunction

class DiscreteHallwayTransition(TransitionFunction):
    def __init__(self,stuck):
        self.stuck_p = stuck        

    def multisample_transition(self,points,action,S=1):
        (N,D) = points.shape
        assert(D == 1)

        
        if (1,) == action.shape:
            multi_action = False
        else:
            multi_action = True
            assert((N,1) == action.shape)
        
        samples = np.empty((S,N,D))

        for s in xrange(S):
            mask = (np.random.rand(N) < self.stuck_p)
            samples[s,mask,:] = points[mask,:]
            if multi_action:
                samples[s,~mask,:] = points[~mask,:]\
                                     + action[~mask,:]
            else:
                samples[s,~mask,:] = points[~mask,:] + action
                

        return samples
        
