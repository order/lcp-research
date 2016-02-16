class FunctionEvaluator(object):
    def evaluate(self,states):
        """
        Evaluates a function at a set of states
        """
        raise NotImplementedError()
        
class InterpolatedFunctionEvaluator(FunctionEvaluator):
    def __init__(self,discretizer,v):
        assert(1 == len(v.shape))
        assert(v.size == discretizer.get_num_nodes())
        self.node_to_cost = v
        self.discretizer = discretizer
        
    def evaluate(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]
            
        M = len(self.node_to_cost)
        (N,d) = states.shape
        
        # Convert state into node dist
        T = self.discretizer.states_to_transition_matrix(states)
        assert((M,N) == T.shape)
        vals = T.T.dot(self.node_to_cost)
        assert((N,) == vals.shape)
        return vals
