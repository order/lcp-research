class ValueFunctionEvaluator(object):
    def evaluate(self,state,action):
        raise NotImplementedError()
        
class BasicValueFunctionEvaluator(ValueFunctionEvaluator):
    def __init__(self,discretizer,v):
        assert(1 == len(v.shape))
        assert(v.size == discretizer.get_num_nodes())
        self.node_to_cost = v
        self.discretizer = discretizer
        
    def evaluate(self,states):
        # Convert state into node dist
        node_dists = self.discretizer.states_to_node_dists(states)        
        
        vals = np.zeros(states.shape[0])
        for (state_id,nd) in node_dists.items():
            for (node_id,w) in nd.items():
                vals[state_id] += self.node_to_cost[node_id] * w                
        return vals