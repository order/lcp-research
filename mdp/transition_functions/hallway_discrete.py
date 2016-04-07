import numpy as np
import scipy as sp
import scipy.sparse as sps

from mdp.transition import TransitionFunction

class DiscreteHallwayTransition(TransitionFunction):
    def __init__(self,N):
        self.num_states = N
        self.sigma = 0.01
        
    def transition(self,points,action,S=1):
        (Np,d) = points.shape
        
        assert(d == 1) # 'points' are just node indices
        assert(np.all(0 <= points)) # +ve
        Ns = self.num_states
        assert(np.all(points < Ns))
        assert(np.sum(np.fmod(points[:,0],1)) < 1e-15) # ints

        assert(1 == len(action.shape))
        assert(1 == action.shape[0]) # just a singleton
        a_id = action[0]

        samples = np.tile(points,(S,1,1))\
                  + a_id + self.sigma*np.random.randn(S,Np,d)
        samples = np.round(samples).astype('i')
        samples = np.mod(samples,self.num_states)
        assert((S,Np,d) == samples.shape)

        return samples
        
