import numpy as np
import scipy.sparse as sps
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
        
    def get_basic_boundary(self):
        raise NotImplementedError()

    def get_actions(self):
        raise NotImplementedError()

    def get_basic_len(self):
        raise NotImplementedError()

    def get_dimension(self):
        raise NotImplementedError()

class ContinuousMDPDiscretizer(MDPDiscretizer):
    """
    A MDP discretizer for a continuous state space.
    Has a distinguished state-remapper responsible for physical evolution
    and a distinguished node-mapper responsible for discretizing non-abstract
    aspects of state-space.
    """
    def __init__(self,physics,basic_mapper,cost_obj,actions,discount):
        self.exception_state_remappers = [] # For domain wrapping
        self.physics = physics # How physical states map to others
        
        self.exception_node_mappers = [] # For stuff like out-of-bounds
        self.basic_mapper = basic_mapper # Essential discretization scheme

        self.drain_states = [] # For implicit "win game" states
        
        self.cost_obj = cost_obj
        
        if (1 == len(actions.shape)):
            actions = actions[:,np.newaxis] # convert to column vector
            
        self.actions = actions
        
        self.num_actions = actions.shape[0]
        self.action_dim = actions.shape[1]

        self.state_dim = self.basic_mapper.get_dimension()

        self.discount = discount


    def get_basic_boundary(self):
        return self.basic_mapper.get_boundary()
        
    def get_basic_lengths(self):
        return self.basic_mapper.get_lengths()

    def __str__(self):
        S = []
        S.append('Physics: {0}'.format(self.physics))
        for remapper in self.exception_state_remappers:
            S.append('\tException: {0}'.format(remapper))
        S.append('Basic Mapper: {0}'.format(self.basic_mapper))
        for mapper in self.exception_node_mappers:
            S.append('\tException: {0}'.format(mapper))
        return '\n'.join(S)

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
        
        Exception nodes are assumed to be NaN
        """
        states = self.basic_mapper.get_node_states()

        # Pad with NaN for any non-physical states
        D = self.basic_mapper.get_dimension()
        N = self.basic_mapper.get_num_nodes()
        
        num_nan_states = self.get_num_nodes() - N
        nans = np.NaN * np.ones((num_nan_states,D))        
        return np.vstack([states,nans])
        
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
        assert(d == self.basic_mapper.get_dimension())
        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            states = remapper.remap(states)
        
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
        assert(2 == len(states.shape))
        (N,d) = states.shape
        assert(d == self.basic_mapper.get_dimension())

        next_states = self.physics.remap(states,action=action)
        for remapper in self.exception_state_remappers:
            next_states = remapper.remap(next_states)
        return next_states

    def states_to_transition_matrix(self,states):
        """
        Maps states to node distributions using all the node mappers
        """       
        assert(2 == len(states.shape))
        (N,d) = states.shape
        assert(d == self.basic_mapper.get_dimension())

        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            states = remapper.remap(states)
        
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
        for a in xrange(self.num_actions):
            action = self.actions[a,:]
            transitions.append(self.build_transition_matrix(action))
            costs.append(self.build_cost_vector(action))
            
        if 'name' not in kwargs:
            kwargs['name'] = 'MDP from Discretizer'
        mdp_obj = mdp.MDP(transitions,
                          costs,
                          self.actions,
                          self.discount,
                          **kwargs)

        return mdp_obj

    def build_drain_mask(self):
        """
        Think about caching this.
        """
        total_nodes = self.get_num_nodes()
        drain_mask = np.ones(total_nodes,dtype=bool)
        drain_mask[self.drain_states] = False
        assert(np.sum(drain_mask) == self.get_num_non_drain_states())
        
        return drain_mask

    def get_num_non_drain_states(self):
        return self.get_num_nodes() - len(self.drain_states)
        
    def build_cost_vector(self,action):
        """
        Build the cost vector for an action
        """

        # Action should be a vector of approp. size
        assert((self.action_dim,) == action.shape)
        
        node_states = self.get_node_states()
        drain_mask = self.build_drain_mask()

        # Get costs of non-drain node states
        costs = self.cost_obj.cost(node_states[drain_mask,:],action)

        # Size check
        non_drain_states = self.get_num_non_drain_states()
        assert((non_drain_states,) == costs.shape)
        
        return costs
    
    def build_transition_matrix(self,action,**kwargs):    
        """
        Builds a transition matrix based on the physics and exceptions
        """
        assert(1 == len(action.shape)) # Should be a vector
        assert(self.action_dim == action.size)
        
        total_nodes = self.get_num_nodes()
        basic_nodes = self.basic_mapper.get_num_nodes()
        assert(total_nodes >= basic_nodes)
        
        # Get the node states, and then use physics to remap them
        node_states = self.basic_mapper.get_node_states()
        assert(2 == len(node_states.shape))
        next_states = self.physics.remap(node_states,action=action)
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


        if len(self.drain_states) > 0:
            # Form the drain mask
            drain_mask = self.build_drain_mask()
            T = T[:,drain_mask]
            T = T[drain_mask,:]
            # Is this really the way to do this?
            # T[mask,mask] seems to do the wrong thing.

        # Size check
        non_drain_nodes = self.get_num_non_drain_states()
        assert((non_drain_nodes,non_drain_nodes) == T.shape)
        assert((T.sum() - non_drain_nodes) / non_drain_nodes <= 1e-12)
        # Basic sanity check: if we just removed columns the sum should be
        # exactly non_drain_nodes. We also removed some rows.
        
        return T

    def add_drain(self,drain):
        """
        Adds a drain node to the discretizer.
        The drain is presented as a node states (e.g. [0,0]).
        """
        assert((self.state_dim,) == drain.shape) # D-vector

        # Get the node distribution for this state
        dist = self.states_to_transition_matrix(drain[np.newaxis,:])
        assert((self.get_num_nodes(),1) == dist.shape)

        # Check that it is a pure state
        dist = dist.tocoo()
        assert(abs(1.0 - dist.max()) < 1e-12) # Max close to one
        assert(abs(1.0 - dist.sum()) < 1e-12) # Sum close to one

        # Get the index of the max element
        idx = dist.data.argmax()
        assert(0 == dist.col[idx])
        assert(abs(1.0 - dist.data[idx]) < 1e-12)
        drain_node_id = dist.row[idx]

        # Expensive check: map back
        states = self.get_node_states()
        assert(np.linalg.norm(drain - states[drain_node_id]) < 1e-12)

        # Add it to the list
        self.drain_states.append(drain_node_id)
               
        
    def pad_with_drain(self,X,pad_value=0.0):
        """
        Take (I,N) matrix and pad it so
        """

        # Well, we also support single vectors
        was_vector = (1 == len(X.shape))
        if was_vector:
            X = X[np.newaxis,:]

        (I,n) = X.shape
        D = len(self.drain_states)
        N = n + D # What we think the length should be
        assert(N == self.get_num_nodes()) # Confirmed
        
        Pad = np.empty((I,N))
        mask = self.build_drain_mask()

        # Mask is True unless it is a drain state
        Pad[:,mask] = X # Non-drain states take their own value
        Pad[:,~mask] = pad_value # Pad drain states with provided value

        # Transform back to vector if was a vector originally.
        if was_vector:
            Pad = Pad[0,:]
        return Pad

    def pad_stack_with_drain(self,X,pad_value=0.0):
        """
        Support for 'stacks', which are the concats of
        the value and flow vectors (primal and dual variables)
        """
        
        was_vector = (1 == len(X.shape))
        if was_vector:
            X = X[np.newaxis,:]

        (I,J) == X.shape
        D = len(self.drain_states)
        A = self.num_actions
        N = self.get_num_nodes()
        padded_size = J + (A+1)*D # Size, plus A+1 pads
        actual = (A+1) * N # What it actually should be
        assert(padded_size == actual)
        assert(J == (A+1)*(N - D)) # Same as above

        Pad = np.empty((I,padded_size))

        for a in xrange(A+1):
            pre_slice = slice(a*(N-D),
                              (a+1)*(N-D)) # Slice of component in X
            post_slice = slice(a*N,
                               (a+1)*N) # Target slice in Pad
            Pad[post_slice] = self.pad_with_drain(X[pre_slice],
                                                  pad_value)
        
        if was_vector:
            Pad = Pad[0,:]
        return Pad       
        
