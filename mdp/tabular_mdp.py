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
                  **kwargs):
        num_states = self.num_states
        A = self.num_actions
        
        state_weights = kwargs.get('state_weights',np.ones(num_states))

        """
        The next two arrays specify what states are to be included
        in the model.
        They're dropped by omission.
        """
        indices = kwargs.get('indices',np.arange(num_states))

        n = indices.size # Number of states used in LCP
        N = (A+1)*n        
        
        # Build the LCP
        q = np.empty(N)
        q[0:n] = -state_weights[indices]

        row = []
        col = []
        data = []
        for a in xrange(A):
            shift = (a+1)*n
            E = self.get_E_matrix(a)
            assert(isinstance(E,sps.csr_matrix))
            E = ((E[indices,:]).tocsc()[:,indices]).tocoo()
            assert((n,n) == E.shape)
            
            row.extend([E.row,E.col + shift])
            col.extend([E.col + shift,E.row])
            data.extend([E.data,-E.data])
            
            q[shift:(shift+n)] = self.costs[a][indices]

        row.append(np.arange(N))
        col.append(np.arange(N))
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

def aggregate_transitions(mdp):
    N = mdp.num_states
    T = sps.csr_matrix((N,N))
    A = len(mdp.transitions)
    for M in mdp.transitions:
        T +=  1.0/float(A) * M
    return T

def find_sinks(transition_matrix):
    T = transition_matrix
    (N,n) = T.shape
    assert(N == n)

    # Get the diagonal
    d = T.diagonal()
    
    # Find those super close to 1.0
    # I.e. only self-transition
    idx = np.where(np.abs(d - 1.0) < 1e-15)[0]
    return idx

def find_unreachable(transition_matrix):
    T = transition_matrix
    (N,n) = T.shape
    assert(N == n)

    # Remove all diagonal entries
    d = T.diagonal()
    D = sps.diags(d,0)
    T -= D

    agg = np.array(T.sum(axis=1)).reshape(-1)
    idx = np.where(np.abs(agg.reshape(-1)) < 1e-15)[0]
    return idx    
    

def find_isolated(mdp,disc):
    cutpoints = disc.get_cutpoints()
    T = aggregate_transitions(mdp)

    if False:
        sinks = find_sinks(T)
        print 'Sinks:'
        for idx in sinks:
            if idx >= disc.num_real_nodes():
                print '[{0}] OOB'.format(idx)
            else:
                print '[{0}] {1}'.format(idx,cutpoints[idx,:])
            
    unreach = find_unreachable(T)
    if False:
        print 'Unreachable:'
        for idx in unreach:
            if idx >= disc.num_real_nodes():
                print '[{0}] OOB'.format(idx)
            else:
                print '[{0}] {1}'.format(idx,cutpoints[idx,:])

    return unreach

def expand_states(mdp,p,d,included_states):
    idx = included_states
    I = idx.size

    # Full shapes    
    n = mdp.num_states
    A = mdp.num_actions
    N = n * (A+1)

    K = I*(A+1)

    print 'p shape',p.shape,(K,)
    assert((K,) == p.shape)
    assert((K,) == d.shape)
    
    block_p = p.reshape((I,(A+1)),order='F')
    block_d = d.reshape((I,(A+1)),order='F')

    idx = included_states

    
    out_mask = np.ones(N)
    out_mask[idx] = 0.0
    nidx = np.where(out_mask == 1.0)[0]
    
    P = np.empty((n,(A+1)))
    P[idx,:] = block_p
    P[nidx,:] = np.nan
    P = P.reshape(-1,order='F')

    D = np.empty((n,(A+1)))
    D[idx,:] = block_d
    D[nidx,:] = np.nan
    D = D.reshape(-1,order='F')

    return (P,D)
