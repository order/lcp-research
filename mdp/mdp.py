import numpy as np
import scipy
import scipy.sparse as sps

import matplotlib.pyplot as plt

import lcp
from utils.parsers import KwargParser

class MDPGenerator(object):
    """
    Abstract class for anything that can generate an MDP
    """
    def build_mdp(self,**kwargs):
        """
        Construct the MDP
        """
        raise NotImplementedError()

class MDP(object):
    """
    MDP object
    """
    def __init__(self,transitions,costs,actions,discount,**kwargs):
        parser = KwargParser()
        parser.add('name','Unnamed')
        parser.add_optional('state_weights')
        parser.add('value_regularization',1e-12)
        parser.add('flow_regularization',1e-12)
        args = parser.parse(kwargs)
        
        self.name = args['name']

        self.discount = discount
        self.transitions = transitions
        self.costs = costs
        self.actions = actions


        
        self.val_reg = args['value_regularization']
        self.flow_reg = args['flow_regularization']
        
        A = len(actions)
        N = costs[0].size
        self.num_actions = A
        self.num_states = N

        assert(len(transitions) == A)
        assert(len(costs) == A)

        # State-weight generation
        if 'state_weights' in args:
            self.state_weights = args['state_weights']
        else:
            # Uniform if not specified
            self.state_weights = np.ones(N)
        assert((N,) == self.state_weights.shape)
        
        # Ensure sizes are consistent
        for i in xrange(A):
            assert((N,) == costs[i].shape)
            assert(not np.any(np.isnan(costs[i])))
            
            assert((N,N) == transitions[i].shape)
            # Stochastic checking removed
            
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        
        I guess we're using a sparse matrix here...
        """
        return sps.eye(self.num_states,format='lil')\
            - self.discount * self.transitions[a].T

    def tolcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        #Build the LCP
        Top = sps.lil_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -self.state_weights
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = sps.hstack((-E,sps.lil_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = sps.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = sps.hstack((Top,E.T))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = sps.vstack((Top,Bottom))

        Reg = sps.lil_matrix((N,N))
        Reg[n:,n:] = self.flow_reg*sps.eye(A*n)
        Reg[:n,:n] = self.val_reg*sps.eye(n)
        
        M = M + Reg

        assert((N,N) == M.shape)
        assert((N,) == q.shape)
        
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
            I_list.append(sps.eye(self.num_states))
        self.B = mdp_skew_assembler(I_list)
        self.C = self.LCP.M - self.B

def mdp_skew_assembler(A_list):
    """
    Builds a skew-symmetric block matrix from a list of squares
    """
    k = len(A_list)
    (n,m) = A_list[0].shape
    for i in xrange(k):
        assert((n,m) == A_list[i].shape)
    
    M = k * m
    # Block = [A_1 ... A_k]    
    Block = sps.hstack(A_list,format='lil')
    assert((n,M) == Block.shape)
    
    # Top = [0 A_1 ... A_k]
    Z_n = sps.lil_matrix((n,n))
    assert(Block.shape[0] == Z_n.shape[0])
    
    Top = sps.hstack([Z_n, Block],format='lil')
    assert((n,M+n) == Top.shape)

    # Bottom = [-Top.T 0]
    Z_M = sps.lil_matrix((M,M))
    assert(Block.shape[1] == Z_M.shape[0])
    Bottom = sps.hstack([-Block.T,Z_M],format='lil')
    assert((M,M+n) == Bottom.shape)

    # Full thing: [[0 Top], [-Top.T 0]]
    SS = sps.vstack([Top,Bottom],format='csr')
    assert((M+n,M+n) == SS.shape)    

    return SS
        
