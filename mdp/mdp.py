import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt

import lcp
from utils.parsers import KwargParser

class MDPBuilder(object):
    def __init__(self,transition_function, # Main physics
                 discretizer, # Maps points to nodes
                 cost_function,
                 actions, # discrete action set
                 discount,
                 samples):

        self.transition_function = transition_function
        self.discretizer = discretizer
        self.cost_function = cost_function
        self.actions = actions
        self.discount = discount
        self.samples = samples

        self.state_remappers = []
        self.node_remappers = []

        self.mean_interp = True
        """
        True to interpolate the mean sample, (probably faster)
        False to average sample interpolation (probably slower, but better)
        """
        
    def build_mdp(self):
        actions = self.actions
        disc = self.discretizer
        (A,action_dim) = actions.shape
        S = self.samples

        points = disc.get_cutpoints()
        (N,state_dim) = points.shape
        assert(N == disc.num_nodes)

        trans_matrices = []
        costs = []
        for a in xrange(A):
            # Costs
            costs[a] = self.cost_function.cost(points,
                                               actions[a,:])

            # Transitions
            samples = self.transition_function.transition(points,
                                                          actions[a,:],
                                                          samples=samples)
            assert((S,N,state_dim) = samples.shape)

            if mean_interp:
                mean_samples = np.mean(samples,axis=0)
                assert((N,state_dim) == mean_samples.shape)
                # Interpolate the mean sample
                T = disc.points_to_index_distributions(mean_sample)
            else:
                T = sps.coo_matrix(shape=(N,N))
                for s in xrange(S):
                    # Interpolate the sample
                    sample_T = disc.points_to_index_distributions(
                        samples[s,:,:])
                    T += 1.0 / float(S) * sample_T # Average the interp.
            trans_matrix.append(T)

            
        return TabularMDP(trans_matrices,
                          costs,
                          actions,
                          discount,
                          ...) 
    
class DiscreteMDP(object):
    pass
   
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
