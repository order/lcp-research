import numpy as np
import scipy.sparse
import mdp
import itertools

class ContinuousMDPDiscretizer(object):
    def __init__(self,physics,basic_mapper,cost_obj,actions):
        self.exception_state_remappers = []
        self.physics = physics
        
        self.exception_node_mappers = []
        self.basic_mapper = basic_mapper
        
        self.cost_obj = cost_obj
        
        self.actions = actions
        
    def get_node_ids(self):
        """
        Return an iterator for all the nodes in the discretization
        """
        node_iters = list(self.basic_mapper.get_nodes())
        for mapper in self.exception_node_mappers:
            node_iters.extend(list(mapper.get_nodes()))
        return node_iters
        
    def get_node_states(self):
        """
        Returns a np.array containing all the node states
        
        Exception nodes are assumed to be NaN
        """
        states = [self.basic_mapper.get_node_states()]
        D = self.basic_mapper.get_dimension()
        for mapper in self.exception_node_mappers:
            states.append(np.NaN * np.ones((mapper.get_num_nodes(),D)))
            
        return np.vstack(states)
        
        
    def add_state_remapper(self,remapper):
        self.exception_state_remappers.append(remapper)
        
    def add_node_mapper(self,mapper):
        self.exception_node_mappers.append(mapper)
        
    def build_mdp(self,**kwargs):
        """
        Build the MDP after all the exceptions have been set up
        
        kwargs is mostly for passing through stuff to the MDP object
        """
        transitions = []
        costs = []
        for a in self.actions:
            transitions.append(self.__build_transition_matrix(a))
            costs.append(self.__build_cost_vector(a))
        if 'name' not in kwargs:
            kwargs['name'] = 'MDP from Discretizer'
        mdp_obj = mdp.MDP(transitions,costs,self.actions,**kwargs)
        return mdp_obj
        
    def __build_cost_vector(self,action):
        """
        Build the cost vector
        """
        node_ids = self.get_node_ids()
        node_states = self.get_node_states()
        return self.cost_obj.cost(node_ids,node_states,action)        
    
    def __build_transition_matrix(self,action,**kwargs):
        """
        Builds a transition matrix based on the physics and exceptions
        """
        
        sparse = kwargs.get('sparse',True)
        
        if not sparse:
            raise NotImplementedError('Not doing dense yet')
        
        # Get the node states, and then use physics to remap them
        node_states = self.basic_mapper.get_node_states()
        next_states = self.physics.remap(node_states,action=action)
        
        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            next_states = remapper.remap(next_states)
        
        # Then map states to node distributions
        dealt_with = set()
        node_mapping = {}
        
        # Deal with the exceptions first
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.states_to_node_dists(next_states,ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
            
        # Then the using the basic remapper
        essential_mapping = self.basic_mapper.states_to_node_dists(next_states,ignore=dealt_with)
        node_mapping.update(essential_mapping)
        
        # All accounted for; no extras
        assert(len(node_mapping) == self.basic_mapper.get_num_nodes())

        # TODO: convert map of node dists into matrix
        total_node_number = self.basic_mapper.get_num_nodes()
        for mapper in self.exception_node_mappers:
            total_node_number += mapper.get_num_nodes()
            
        P = scipy.sparse.dok_matrix((total_node_number,total_node_number))
            
        for (source_node, next_node_dist) in node_mapping.items():
            for (next_node,weight) in next_node_dist.items():
                P[next_node,source_node] = weight
                
        if sparse:
            P = P.tocsr()
            
        return P
        