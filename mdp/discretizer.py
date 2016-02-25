import numpy as np
import scipy.sparse as sps
import mdp
import node_mapper
import itertools
import matplotlib.pyplot as plt
import linalg

import utils
from utils.parsers import KwargParser

class MDPDiscretizer(mdp.MDPBuilder):
    """
    Abstract class for classes for describing an MDP builder
    by discretizing a continuous MDP
    """
    def states_to_node_dists(self,states,**kwargs):
        """
        Takes an arbitrary Nxd ndarray and maps it to
        each of the rows to a distribution over nodes
        """
        raise NotImplementedError()

    def get_num_nodes(self):
        """
        Return the number of nodes in the discretizer.

        Some of the nodes may be non-physical, so different than
        np.prod(self.get_basic_lengths())
        """
        raise NotImplementedError()

    def get_special_node_indices(self):
        """
        Return the indices of any `special' or non-physical nodes
        Typically the physical nodes are [0,N], and all the non-physical
        Nodes follow.
        """
        raise NotImplementedError()        
    
    def get_actions(self):
        """
        Gets the list of actions. Each action will be kD, where k is
        the dimension of the control space.
        """
        raise NotImplementedError()

    def get_num_actions(self):
        """
        Return the number of actions
        """
        raise NotImplementedError()    
        
    def get_basic_boundary(self):
        """
        Returns a list of pairs, with the min and max along each dimension.
        So [(-1,1),(-5,5)] could be the boundary for a problem, indicating
        that the problem is 2D with a rectangular geometry [-1,1] x [-5,5]
        """
        raise NotImplementedError()

    def get_basic_lengths(self):
        """
        Returns a tuple with the number of states in each dimension,
        so (4,5) would indicate that there are 4 discrete states in the
        discretization of the 1st dimension, and 5 in the second
        """
        raise NotImplementedError()

    def get_dimension(self):
        """
        Return the number of physical dimensions in the problem,
        e.g. 2 for the 1D double integrator (position and velocity)
        """
        raise NotImplementedError()

class ContinuousMDPDiscretizer(MDPDiscretizer):
    """
    A MDP discretizer for a continuous state space.
    Has a distinguished state-remapper responsible for physical evolution
    and a distinguished node-mapper responsible for discretizing non-abstract
    aspects of state-space.
    """
    def __init__(self, problem,
                 basic_mapper,
                 actions):

        self.problem = problem
        
        self.exception_node_mappers = [] # For stuff like out-of-bounds
        self.basic_mapper = basic_mapper # Essential discretization scheme
                
        if (1 == len(actions.shape)):
            actions = actions[:,np.newaxis] # convert to column vector
        assert(self.problem.action_dim == actions.shape[1])
        self.actions = actions # Discrete actions
        
        self.num_actions = actions.shape[0]

        self.state_dim = self.problem.dimension
        assert(self.state_dim == self.basic_mapper.get_dimension())
        assert(self.problem.get_boundary() == self.basic_mapper.get_boundary())

    def get_basic_boundary(self):
        return self.basic_mapper.get_boundary()
        
    def get_basic_lengths(self):
        return self.basic_mapper.get_lengths()

    def __str__(self):
        raise NotImplementedError()

    def get_num_actions(self):
        return self.num_actions

    def get_dimension(self):
        return self.state_dim
        
    def get_node_ids(self):
        """
        Return an all the node ids in the discretization
        """
        node_set = set(self.basic_mapper.get_node_ids())
        for mapper in self.exception_node_mappers:
            node_set |= set(mapper.get_node_ids())
        return node_set
        
    def get_node_states(self):
        """
        Returns a np.array containing all the node states
        
        Special nodes are NaN
        """
        states = self.basic_mapper.get_node_states()

        # Pad with NaN for any non-physical states
        D = self.basic_mapper.get_dimension()
        N = self.basic_mapper.get_num_nodes()

        # NaN always at the end.
        num_nan_states = self.get_num_nodes() - N
        nans = np.NaN * np.ones((num_nan_states,D))        
        return np.vstack([states,nans])

    def get_special_node_indices(self):
        """
        Return the indices of the special nodes
        """
        N = self.basic_mapper.get_num_nodes()
        count = 0
        for mapper in self.exception_node_mappers:
            count += mapper.get_num_nodes()
        return range(N,N+count)
        
    def get_num_nodes(self):
        """
        Count of nodes, including basic and special nodes
        """
        count = self.basic_mapper.get_num_nodes()
        for mapper in self.exception_node_mappers:
            count += mapper.get_num_nodes()
        return count

    def states_to_node_dists(self,states):
        """
        Maps states to node distributions using all the node mappers
        """
        assert(2 == len(states.shape))
        (N,d) = states.shape
        assert(d == self.basic_mapper.get_dimension())
        
        # Map states to node distributions
        dealt_with = set() # Handled by an earlier mapper
        node_mapping = {} # Partial mapping so far
        
        # Deal with the exceptions firts
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.\
                              states_to_node_dists(states,\
                                                   ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
            
        # Then the using the basic remapper
        essential_mapping = self.basic_mapper.\
                            states_to_node_dists(states,\
                                                 ignore=dealt_with)
        node_mapping.update(essential_mapping)
        
        # All nodes are dealt with
        if len(node_mapping) != N:
            for i in xrange(N):
                if i not in node_mapping:
                    print "!! Missing state {0}: {1}".format(i, states[i,:])
        
        return node_mapping 
        
    def remap_states(self,states,action):
        return problem.next_states(states,
                                   action,
                                   uniform_action=True)

    def states_to_transition_matrix(self,states):
        """
        Maps states to node distributions using all the node mappers
        """       
        assert(2 == len(states.shape))
        (N,d) = states.shape
        assert(d == self.basic_mapper.get_dimension())
        
        # Then map states to node distributions
        dealt_with = set() # Handled by an earlier mapper
        node_mapping = {} # Partial mapping so far
        
        # Deal with the exceptions first
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.\
                              states_to_node_dists(states,\
                                                   ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
            
        # Then use the basic remapper
        total_nodes = self.get_num_nodes()
        num_basic_nodes = self.basic_mapper.get_num_nodes()
        assert(total_nodes >= num_basic_nodes)
        
        # Convert the basic states to a transition matrix
        T = self.basic_mapper.\
            states_to_transition_matrix(states,\
                                        shape=(total_nodes,N),\
                                        ignore=dealt_with)
        assert((total_nodes,N) == T.shape)        
        T = T.tocsr()
        
        # Add any exceptions
        IJ = []
        Data = []
        if node_mapping:
            I = 0
            for (state_id,nd) in node_mapping.items():                
                # Write the 
                for (node_id,w) in nd.items():
                    IJ.append([node_id,state_id])
                    Data.append(w)
            Data = np.array(Data)
            IJ = np.array(IJ).T
            E = Data.size
            assert((E,) == Data.shape)
            assert((2,E) == IJ.shape)
            T = T + sps.csr_matrix((Data,IJ),shape=(total_nodes,N))        
        return T
        
    def add_node_mapper(self,mapper):
        self.exception_node_mappers.append(mapper)
        
    def build_mdp(self):
        """
        Build the MDP after all the exceptions have been set up        
        """

        # Build the transitions, costs, and so forth
        transitions = []
        costs = []
        utils.banner('Checking self-transitioning:')
        for a in xrange(self.num_actions):
            action = self.actions[a,:]
            T= self.build_transition_matrix(action)
            transitions.append(T)
            stp = 100.0*linalg.trace(T) / T.sum()
            print '\tAction {0}: {1}%'.format(a,stp)
            costs.append(self.build_cost_vector(action))

        weight = self.build_weight_vector()
            
        mdp_obj = mdp.MDP(transitions,
                          costs,
                          self.actions,
                          self.problem.discount,
                          weight,
                          name='MDP from Discretizer')

        return mdp_obj
        
    def build_cost_vector(self,action):
        """
        Build the cost vector for an action
        """

        # Action should be a vector of approp. size
        assert((self.problem.action_dim,) == action.shape)
        
        node_states = self.get_node_states()

        # Get costs of non-drain node states
        costs = self.problem.cost_obj.evaluate(node_states,
                                               action=action)

        # Size check
        assert((self.get_num_nodes(),) == costs.shape)
        
        return costs

    def build_weight_vector(self):
        """
        Build the weight vector for an action
        """        
        node_states = self.get_node_states()

        # Get costs of non-drain node states
        weight = self.problem.weight_obj.evaluate(node_states)

        # Size check
        assert((self.get_num_nodes(),) == weight.shape)
     
        return weight
    
    def build_transition_matrix(self,action,**kwargs):    
        """
        Builds a transition matrix based on the physics and exceptions
        """
        assert(1 == len(action.shape)) # Should be a vector
        assert(self.problem.action_dim == action.size)
        
        total_nodes = self.get_num_nodes()
        basic_nodes = self.basic_mapper.get_num_nodes()
        assert(total_nodes >= basic_nodes)
        
        # Get the node states, and then use physics to remap them
        node_states = self.basic_mapper.get_node_states()
        assert(2 == len(node_states.shape))
        actions = np.tile(action,(basic_nodes,1))
        next_states = self.problem.next_states(node_states,
                                               actions)
        assert(node_states.shape == next_states.shape)
        
        # Then get the node mappings for all next states
        T = self.states_to_transition_matrix(next_states)
        assert((total_nodes,basic_nodes) == T.shape)
        if total_nodes > basic_nodes:
            # Add non-physical nodes as a set of new columns
            T = sps.hstack([T,sps.lil_matrix((total_nodes,
                                              total_nodes-basic_nodes))])
        T = T.tocsr()            
        
        # Make sink nodes transition purely to themselves.
        A = sps.lil_matrix((total_nodes,total_nodes))
        for mapper in self.exception_node_mappers:
            nid = mapper.sink_node
            A[nid,nid] = 1.0
        T = T + A.tocsr()

        # Size check
        assert((total_nodes,total_nodes) == T.shape)
        assert((T.sum() - total_nodes) / total_nodes <= 1e-12)
        
        return T
