import numpy as np
import scipy.sparse
import mdp
import node_mapper
import itertools

class MDPDiscretizer(object):
    """
    Abstract class defining how states are mapped to discrete nodes.
    Usually a collection of StateRemappers and NodeMappers
    """
    def states_to_node_dists(self,states,**kwargs):
        """
        Takes an arbitrary Nxd ndarray and maps it to
        each of the rows to a distribution over nodes
        """
        raise NotImplementedError()
        
    def build_mdp(self,**kwargs):
        """
        Construct the MDP
        """
        raise NotImplementedError()


class ContinuousMDPDiscretizer(MDPDiscretizer):
    """
    A MDP discretizer for a continuous state space.
    Has a distinguished state-remapper responsible for physical evolution
    and a distinguished node-mapper responsible for discretizing non-abstract
    aspects of state-space.
    """
    def __init__(self,physics,basic_mapper,cost_obj,actions):
        self.exception_state_remappers = []
        self.physics = physics
        
        self.exception_node_mappers = []
        self.basic_mapper = basic_mapper
        
        self.cost_obj = cost_obj
        
        self.actions = actions   

    def __str__(self):
        S = []
        S.append('Physics: {0}'.format(self.physics))
        for remapper in self.exception_state_remappers:
            S.append('\tException: {0}'.format(remapper))
        S.append('Basic Mapper: {0}'.format(self.basic_mapper))
        for mapper in self.exception_node_mappers:
            S.append('\tException: {0}'.format(mapper))
        return '\n'.join(S)
       
        
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
        
        Exception nodes are assumed to be NaN
        """
        states = [self.basic_mapper.get_node_states()]
        D = self.basic_mapper.get_dimension()
        for mapper in self.exception_node_mappers:
            states.append(np.NaN * np.ones((mapper.get_num_nodes(),D)))            
        return np.vstack(states)
        
    def get_num_nodes(self):
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
        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            states = remapper.remap(states)
        
        # Then map states to node distributions
        dealt_with = set() # Handled by an earlier mapper
        node_mapping = {} # Partial mapping so far
        
        # Deal with the exceptions first
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.states_to_node_dists(states,ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
            
        # Then the using the basic remapper
        essential_mapping = self.basic_mapper.states_to_node_dists(states,ignore=dealt_with)
        node_mapping.update(essential_mapping)
        
        # All nodes are dealt with
        if len(node_mapping) != N:
            for i in xrange(N):
                if i not in node_mapping:
                    print "!! Missing state {0}: {1}".format(i, states[i,:])
        
        return node_mapping 

    def states_to_transition_matrix(self,states):
        """
        Maps states to node distributions using all the node mappers
        """       
        
        assert(2 == len(states.shape))
        (N,d) = states.shape
        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            states = remapper.remap(states)
        
        # Then map states to node distributions
        dealt_with = set() # Handled by an earlier mapper
        node_mapping = {} # Partial mapping so far
        
        # Deal with the exceptions first
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.states_to_node_dists(states,ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
            
        # Then the using the basic remapper
        T = self.basic_mapper.states_to_transition_matrix(states,ignore=dealt_with)
        assert(N == T.shape[1])
        
        T = T.tolil() # Make sure in list format
        for (state_id,nd) in node_mapping:
            assert(not np.any(T[:,sid]))
            for (node_id,w) in nd.items():
                T[node_id,state_id] = w
        
        return T
        
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
            transitions.append(self.build_transition_matrix(a))
            costs.append(self.build_cost_vector(a))
        if 'name' not in kwargs:
            kwargs['name'] = 'MDP from Discretizer'
        if 'discount' not in kwargs:
            kwargs['discount'] = 0.99
        mdp_obj = mdp.MDP(transitions,costs,self.actions,**kwargs)
        return mdp_obj
        
    def build_cost_vector(self,action):
        """
        Build the cost vector for an action
        """
        node_states = self.get_node_states()
        return self.cost_obj.cost(node_states,action)        
    
    def build_transition_matrix(self,action,**kwargs):
    
        """
        Builds a transition matrix based on the physics and exceptions
        """
        
        N = self.get_num_nodes()
        sparse = kwargs.get('sparse',True)
        
        if not sparse:
            raise NotImplementedError('Not doing dense yet')
        
        # Get the node states, and then use physics to remap them
        node_states = self.basic_mapper.get_node_states()
        next_states = self.physics.remap(node_states,action=action)
        
        # Then get the node mappings for all next states
        T = self.states_to_transition_matrix(next_states)
        assert((N,N) == T.shape)
        
        # Make sink nodes transition purely to themselves.
        for mapper in self.exception_node_mappers:
            nid = mapper.sink_node
            T[nid,nid] = 1.0
            
        assert(abs(T.sum() - N) < 1e-9) # Coarse check that its stochastic
            
        return T.tocsr()
        