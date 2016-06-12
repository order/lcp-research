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

    def build_lcp(self,
                  val_reg=0.0,
                  flow_reg=1e-15,
                  state_weights=None):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n

        if not state_weights:
            state_weights = np.ones(n)

        # Build the LCP
        use_lil = True # COO method faster
        q = np.empty(N)
        q[0:n] = -state_weights

        row = []
        col = []
        data = []
        for a in xrange(A):
            shift = (a+1)*n
            E = self.get_E_matrix(a).tocoo()
            row.extend([E.row,E.col + shift])
            col.extend([E.col + shift,E.row])
            data.extend([E.data,-E.data])
            q[shift:(shift+n)] = self.costs[a]

        row.extend([np.arange(n),np.arange(n,N)])
        col.extend([np.arange(n),np.arange(n,N)])
        data.extend([val_reg*np.ones(n),
                     flow_reg*np.ones(A*n)])
        row = np.concatenate(row)
        col = np.concatenate(col)
        data = np.concatenate(data)
        M = sps.coo_matrix((data,(row,col)),
                           shape=(N,N))

        name = 'LCP from {0} MDP'.format(self.name)
        return lcp.LCPObj(M,q,name=name)

def add_drain(mdp,disc,state,cost):

    (N,) = state.shape
        
    dist = disc.points_to_index_distributions(state[np.newaxis,:])
    (G,n) = dist.shape
    assert(n == 1)

    max_prob = -1
    node_id = -1

    assert(2 == len(dist.indptr))
    for i in xrange(dist.nnz):
        if dist.data[i] > max_prob:
            max_prob = dist.data[i]
            node_id = dist.indices[i]
    assert(max_prob > 1 - 1e-6)

    elem = sps.dok_matrix((G,1))
    elem[node_id,0] = 1

    for i in xrange(mdp.num_actions):
        T = mdp.transitions[i]
        assert(isinstance(T,sps.csc_matrix))
        T.data[T.indptr[node_id]:T.indptr[node_id+1]] = 0
        T.eliminate_zeros()
        mdp.costs[i][node_id] += cost / (1.0 - mdp.discount)

        assert(mdp.transitions[i].dot(elem).sum() < 1e-12)
