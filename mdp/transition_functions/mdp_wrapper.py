import numpy as np
import scipy as sp
import scipy.sparse as sps

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

class MDPTransitionWrapper(object):
    def __init__(self,transitions):
        self.transitions = transitions
        self.num_states = transitions[0].shape[0]
        
    def transition(self,points,action,S=1):
        (Np,d) = points.shape
        assert(d == 1)
        Ns = self.num_states
        
        row = points[:,0]
        col = np.arange(Np)
        data = np.ones(Np)
        X = sps.coo_matrix((data,(row,col)),shape=(Ns,Np))
        
        Post = self.transitions[action].dot(X)
        assert((Ns,Np) == Post.shape)
        
        Samples = sample_matrix(Post,S)

        Samples = (Samples.T)[:,:,np.newaxis]
        assert((S,Np,d) == Samples.shape)
        return Samples
        
