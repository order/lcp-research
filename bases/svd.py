import numpy as np
import scipy.sparse as sps
import bases
from utils.parsers import KwargParser

class SVDBasis(object):
    def __init__(self,K):
        self.K = K
        
    def generate_basis(self,points=None,**kwargs):
        
        mdp_obj = kwargs['mdp_obj']
        discretizer = kwargs['discretizer']

        A = mdp_obj.num_actions
        N = mdp_obj.num_states
        K = self.K

        Es = [mdp_obj.get_action_matrix(a) for a in xrange(A)]
        E_stack = sps.vstack(Es)

        [U,S,Vt] = sps.linalg.svds(E_stack,
                                   k=K,
                                   tol=1e-8,
                                   maxiter=1e4)
        print U.shape
        print (N,K)
        assert((N,K) == U.shape)
        assert((K,N*A) == Vt.shape)
        Vt = np.reshape(Vt,(K,A,N))
        V = np.rollaxis(Vt,0,2)
        assert((A,N,K) == V.shape)

        for a in xrange(A):
            print np.linalg.norm(Es[a] - U.dot(S).dot(Vt[:,a,:]))

        return [U,V]
