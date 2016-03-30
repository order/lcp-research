import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt

from tabular_mdp import TabularMDP
import lcp
from utils.parsers import KwargParser

class MDPBuilder(object):
    def __init__(self,
                 transition_function, # Main physics function
                 discretizer, # Maps points to nodes
                 state_remappers, # List of special case remappers
                 cost_function,
                 actions, # discrete action set
                 discount,
                 num_samples,
                 mean_interp=True):

        self.transition_function = transition_function
        self.discretizer = discretizer
        self.state_remappers = state_remappers
        self.cost_function = cost_function
        self.actions = actions
        self.discount = discount
        
        self.num_samples = num_samples
        self.mean_interp = mean_interp
        """
        True to interpolate the mean sample, (probably faster)
        False to average sample interpolation (probably slower, but better)
        """
        
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

        trans_fn = self.transition_function
        cost_fn = self.cost_function
        trans_matrices = []
        costs = []
        for a in xrange(A):
            # Costs
            c = cost_fn.cost(points,
                             actions[a,:])
            costs.append(c)
            
            # Transitions
            samples = trans_fn.transition(points,
                                          actions[a,:],
                                          samples=S)
            assert((S,N,state_dim) == samples.shape)

            # Remap
            for remapper in self.state_remappers:
                for s in xrange(S):
                    samples[s,:,:] = remapper.remap(
                        samples[s,:,:])


            # Discretize
            if self.mean_interp:
                mean_samples = np.mean(samples,axis=0)
                assert((N,state_dim) == mean_samples.shape)
                # Interpolate the mean sample
                T = disc.points_to_index_distributions(
                    mean_sample)
            else:
                T = sps.coo_matrix((N,N))
                for s in xrange(S):
                    # Interpolate the sample
                    sample_T = disc.points_to_index_distributions(
                        samples[s,:,:])
                    T = T + 1.0 / float(S) * sample_T # Average the interp.
            trans_matrices.append(T)

            
        return TabularMDP(trans_matrices,
                          costs,
                          actions,
                          self.discount)
   



