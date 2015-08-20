class Discretizer(object):
    def __init__(self,physics,basic_mapper):
        self.exception_state_remappers = []
        self.physics = physics
        
        self.exception_node_mappers = []
        self.basic_mapper = basic_mapper
    
    def build_transition_matrix(self,action):
        node_states = self.basic_disc.get_node_states()
        next_step = self.physics.remap(node_states,action=action)
        
        # First remap any states (velocity cap, etc.)
        for remapper in self.exception_state_remappers:
            next_step = remapper.remap(next_step)
        
        # Then map states to node distributions
        dealt_with = set()
        node_mapping = {}
        
        # Deal with the exceptions first
        for mapper in self.exception_node_mappers:
            partial_mapping = mapper.states_to_node_dists(next_step,ignore=dealt_with)
            node_mapping.update(partial_mapping)
            dealt_with |= set(partial_mapping.keys())
        essential_mapping = self.basic_mapper.states_to_node_dists(next_step,ignore=dealt_with)
        node_mapping.update(essential_mapping)
        
        # All accounted for; no extras
        assert(len(node_mapping) == self.basic_disc.get_num_nodes())

        # TODO: convert map of node dists into matrix
        total_node_number = self.basic_disc.get_num_nodes()
        for mapper in self.exception_node_mappers:
            total_node_number += mapper.get_num_nodes()
            
        P = np.zeros(total_node_number,total_node_number)
        # TODO FINISH
        raise NotImplementedError('Have not finished yet')