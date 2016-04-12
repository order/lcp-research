import numpy as np
import scipy.sparse as sps

from tabular_mdp import TabularMDP

class MDPBuilder(object):
    def __init__(self,
                 generative_model,
                 discretizer, # Maps points to nodes
                 actions, # discrete action set
                 discount,
                 num_samples):

        self.gen_model = generative_model
        self.discretizer = discretizer
        self.actions = actions
        self.discount = discount
        self.num_samples = num_samples
        
    def build_mdp(self):
        # Shorthand
        actions = self.actions
        disc = self.discretizer
        (A,action_dim) = actions.shape
        S = self.num_samples

        # Get tates for each cutpoint
        points = disc.get_cutpoints() 
        (N,state_dim) = points.shape
        assert(N == disc.num_nodes)

        model = self.gen_model
        trans_matrices = []
        costs = []
        for a in xrange(A):
            # Costs
            action = actions[a,:]
            (samples,cost) = model.multisample_next(states,
                                                    action,S)
            assert((S,N,state_dim) == samples.shape)

            costs.append(cost)

            T = sps.coo_matrix((N,N))
            for s in xrange(S):
                # Interpolate the sample
                sample_T = disc.points_to_index_distributions(
                    samples[s,:,:])
                T = T + 1.0 / float(S) * sample_T
                # Average the interp.
            trans_matrices.append(T)

            
        return TabularMDP(trans_matrices,
                          costs,
                          actions,
                          self.discount)
   



