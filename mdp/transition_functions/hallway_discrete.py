import numpy as np
import scipy as sp
import scipy.sparse as sps

from mdp.transition import TransitionFunction

class DiscreteHallwayTransition(TransitionFunction):
    def __init__(self,N,D):
        self.num_states = N
        self.dim = D
        
    def multisample_transition(self,points,actions,S=1):
        (Np,Dp) = points.shape                
        (Na,Da) = actions.shape
        assert(Np == Na)
        Ns = self.num_states

        assert(2 == len(action.shape))
        

        samples = np.tile(points,(S,1,1))\
                  + a_id + self.sigma*np.random.randn(S,Np,d)
        samples = np.round(samples).astype('i')
        samples = np.mod(samples,self.num_states)
        assert((S,Np,d) == samples.shape)

        return samples
        
