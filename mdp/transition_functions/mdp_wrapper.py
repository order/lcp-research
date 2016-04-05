import numpy as np
import scipy as sp
import scipy.sparse as sps

from mdp.transition import TransitionFunction

def sample_matrix(P,S):
    """
    P is column stochastic
    Sample from each of the M cols S times
    """
    (M,N) = P.shape
    Samples = np.empty((N,S))
    for i in xrange(N):
        p = P.getcol(i).tocoo()
        Samples[i,:] = np.random.choice(p.row,
                                        size=S,
                                        p=p.data)
    return Samples

class MDPTransitionWrapper(TransitionFunction):
    def __init__(self,transitions):
        self.transitions = transitions
        self.num_states = transitions[0].shape[0]
        
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
        assert(a_id % 1 < 1e-15) # int
        a_id = int(a_id)
        
        row = points[:,0].astype('i')
        col = np.arange(Np)
        data = np.ones(Np)
        X = sps.coo_matrix((data,(row,col)),shape=(Ns,Np))
        
        post = self.transitions[a_id].dot(X)
        assert((Ns,Np) == post.shape)
        
        samples = sample_matrix(post,S)

        samples = (samples.T)[:,:,np.newaxis]
        assert((S,Np,d) == samples.shape)
        return samples
        
