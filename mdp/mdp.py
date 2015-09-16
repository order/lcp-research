import numpy as np
import scipy
import matplotlib.pyplot as plt
import lcp

class MDP(object):
    """
    MDP object
    """
    def __init__(self,transitions,costs,actions,**kwargs):
        self.discount = kwargs.get('discount',0.99)
        self.transitions = transitions
        A = len(actions)
        N = costs[0].shape[0]
        self.actions = actions
        self.name = kwargs.get('name','Unnamed')
        self.num_actions = A
        self.num_states = N
        
        self.costs = costs

        assert(len(transitions) == A)
        assert(len(costs) == A)
        
        # Ensure sizes are consistent
        for i in xrange(A):
            assert(costs[i].size == N)
            assert(not np.any(np.isnan(costs[i])))
            
            assert(transitions[i].shape[0] == N)
            assert(transitions[i].shape[1] == N)
            assert(abs(transitions[i].sum() - N) <= 1e-6)
            
        
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        
        I guess we're using a sparse matrix here...
        """
        return np.eye(self.num_states) - self.discount * self.transitions[a].T

    def tolcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        Top = scipy.sparse.coo_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -np.ones(n)
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = scipy.sparse.hstack((-E,scipy.sparse.coo_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = scipy.sparse.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = scipy.sparse.hstack((Top,E.T))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = scipy.sparse.vstack((Top,Bottom))
        return lcp.LCPObj(M,q,name='LCP from {0} MDP'.format(self.name))

    def __str__(self):
        return '<{0} with {1} actions and {2} states>'.\
            format(self.name, self.num_actions, self.num_states)
        
        
class MDPValueIterSplitter(object):
    """
    Builds an LCP based on an MDP, and split it
    according to value-iteration based (B,C)
    """
    def __init__(self,MDP,**kwargs):
        self.MDP = MDP
        self.num_actions = self.MDP.num_actions
        self.num_states = self.MDP.num_states
        
        # Builds an LCP based on an MPD
        self.LCP = lcp.MDPLCPObj(MDP)
        self.value_iter_split()      
    
    def update(self,v):
        """
        Builds a q-vector based on current v
        """
        q_k = self.LCP.q + self.C.dot(v)
        return (self.B,q_k)    

    def value_iter_split(self):
        """
        Creates a simple LCP based on value-iteration splitting:
        M = [[0 I I], [-I 0 0], [-I 0 0]]
        """
        I_list = []
        P_list = []
        # Build the B matrix
        for i in xrange(self.num_actions):
            I_list.append(scipy.sparse.eye(self.num_states))
        self.B = mdp_skew_assembler(I_list)
        self.C = self.LCP.M - self.B

def mdp_skew_assembler(A_list):
    """
    Builds a skew-symmetric block matrix from a list of squares
    """
    A = len(A_list)
    (n,m) = A_list[0].shape
    assert(n == m) # Square
    N = (A+1)*n
    M = scipy.sparse.lil_matrix((N,N))
    for i in xrange(A):
        I = xrange(n)
        J = xrange((i+1)*n,(i+2)*n)
        M[np.ix_(I,J)] = A_list[i]
        M[np.ix_(J,I)] = -A_list[i]
    return M.tocsr()
        