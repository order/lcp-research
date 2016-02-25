import numpy as np
import scipy.sparse as sps
import bases
from utils.parsers import KwargParser

import matplotlib.pyplot as plt

class SVDBasis(object):
    def __init__(self,K):
        self.K = K
        
    def generate_basis(self,points=None,**kwargs):
        
        mdp_obj = kwargs['mdp_obj']
        discretizer = kwargs['discretizer']

        A = mdp_obj.num_actions
        N = mdp_obj.num_states
        K = self.K
        #K = 15
        
        Es = [mdp_obj.get_action_matrix(a) for a in xrange(A)]
        E_stack = sps.hstack(Es)
        #E_stack = mdp_obj.get_action_matrix(1)

        [U,S,Vt] = sps.linalg.svds(E_stack,
                                   k=K,
                                   which='LM',
                                   tol=1e-10,
                                   maxiter=1e5)
        
        assert((N,K) == U.shape)
        #Vt = np.hstack([Vt]*A)
        assert((K,N*A) == Vt.shape)        
        Vt = np.reshape(Vt,(K,A,N),order='C')
        Vs = []
        for a in xrange(A):
            V = Vt[:,a,:].T
            Vs.append(V)
        return [U]+Vs
