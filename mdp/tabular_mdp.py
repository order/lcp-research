import numpy as np
import scipy as sp
import scipy.sparse as sps
import linalg

import lcp

class TabularMDP(object):
    """
    MDP object assuming a discrete state-space with tabular representation.
    """
    def __init__(self,transitions,
                 costs,
                 actions,
                 discount):
        self.transitions = [T.tocsc() for T in transitions]
        self.costs = costs
        self.actions = actions
        self.discount = discount
     
        self.name = 'Unnamed'

        A = len(actions)
        N = costs[0].size
        self.num_actions = A
        self.num_states = N

        assert(len(transitions) == A)
        assert(len(costs) == A)
        
        # Ensure sizes are consistent
        for i in xrange(A):
            assert((N,) == costs[i].shape)
            assert(not np.any(np.isnan(costs[i])))            
            assert((N,N) == self.transitions[i].shape)
            assert(isinstance(self.transitions[i],sps.csc_matrix))
            
    def get_E_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a
        """
        assert(isinstance(self.transitions[a],sps.spmatrix))
        return sps.eye(self.num_states,format='lil')\
            - self.discount * self.transitions[a]

    def get_value_residual(self,v):
        N = self.num_states
        A = self.num_actions
        
        comps = np.empty((N,A))
        gamma = self.discount
        for a in xrange(A):
            c = self.costs[a]
            Pt = self.transitions[a].T
            comps[:,a] = c + gamma * Pt.dot(v)

        res = v - np.amin(comps,axis=1)
        assert((N,) == res.shape)
        return res
    
    def aggregate_transitions(self):
        N = self.num_states
        T = sps.csr_matrix((N,N))
        A = len(self.transitions)
        for M in self.transitions:
            T +=  1.0/float(A) * M
        return T
