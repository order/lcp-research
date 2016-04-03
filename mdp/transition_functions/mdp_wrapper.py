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
        assert(d == 1)
        assert(np.all(0 <= points))
        Ns = self.num_states
        assert(np.all(points < Ns))

        assert(np.sum(np.fmod(points[:,0],1)) <1e-15)
        row = points[:,0].astype('i')
        col = np.arange(Np)
        data = np.ones(Np)
        X = sps.coo_matrix((data,(row,col)),shape=(Ns,Np))
        
        post = self.transitions[action].dot(X)
        assert((Ns,Np) == post.shape)
        
        samples = sample_matrix(post,S)

        samples = (samples.T)[:,:,np.newaxis]
        assert((S,Np,d) == samples.shape)
        return samples
        
