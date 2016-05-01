import numpy as np
import scipy as sp
import scipy.sparse as sps
import linalg

from mdp.transition import TransitionFunction

class DiscreteHallwayTransition(TransitionFunction):
    def __init__(self,stuck,nodes):
        self.stuck_p = stuck
        self.nodes = nodes

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
            if multi_action:
                samples[s,~mask,:] = points[~mask,:]\
                                     + action[~mask,:]
            else:
                samples[s,~mask,:] = points[~mask,:] + action
            flips = linalg.rademacher(shape=(mask.sum(),1))
            samples[s,mask,:] = points[mask,:]\
                                + flips

        samples = np.mod(samples,self.nodes)
        assert((S,N,D) == samples.shape)
        assert(not np.any(samples < 0))
        assert(not np.any(samples >= self.nodes))

        return samples
        
