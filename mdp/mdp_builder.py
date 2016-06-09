import numpy as np
import scipy.sparse as sps

from tabular_mdp import TabularMDP

import matplotlib.pyplot as plt

class MDPBuilder(object):
    def __init__(self,
                 problem,
                 discretizer, # Maps points to nodes
                 actions, # discrete action set
                 num_samples):

        self.problem = problem
        self.discretizer = discretizer
        self.actions = actions
        self.num_samples = num_samples
        
    def build_mdp(self):
        # Shorthand
        actions = self.actions
        disc = self.discretizer

        
        (A,action_dim) = actions.shape
        S = self.num_samples

        # Get all real cutpoints
        N = disc.num_nodes() # Including oob
        points = disc.get_cutpoints() 
        (n,state_dim) = points.shape
        assert(n == disc.num_real_nodes())

        num_oob = disc.num_oob()
        assert(num_oob == 2*state_dim)

        model = self.problem.gen_model
        trans_matrices = []
        costs = []
        for a in xrange(A):
            # Costs
            action = actions[a,:]
            (samples,cost) = model.multisample_next(points,
                                                    action,S)
            print cost.shape
            assert((S,n,state_dim) == samples.shape)
            assert((n,) == cost.shape)
            exp_cost = np.empty(N)
            exp_cost[:n] = cost
            exp_cost[n:] = model.oob_costs
            costs.append(exp_cost)

            # Collect the physical transitions
            # from the n real nodes to any of the
            # N nodes (including oob)
            T = sps.coo_matrix((N,n))
            for s in xrange(S):
                # Interpolate the sample
                sample_T = disc.points_to_index_distributions(
                    samples[s,:,:])
                T = T + 1.0 / float(S) * sample_T
                # Average the interp.
            T = T.tocoo()
            nnz = T.nnz
            data = np.empty(nnz + num_oob)
            col = np.empty(nnz + num_oob)
            row = np.empty(nnz + num_oob)

            data[:nnz] = T.data
            col[:nnz] = T.col
            row[:nnz] = T.row

            data[nnz:] = np.ones(num_oob)
            col[nnz:] = np.array(disc.get_oob_indices())
            row[nnz:] = np.array(disc.get_oob_indices())
            # Expand with
            T = sps.coo_matrix((data,(row,col)),shape=(N,N))
            
            trans_matrices.append(T)
            
        return TabularMDP(trans_matrices,
                          costs,
                          actions,
                          self.problem.discount)
   



