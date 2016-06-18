import state_functions
import numpy as np

class CostFunction(object):
    def cost(self,states,action):
        raise NotImplementedError()
    def single_cost(self,state,action):
        assert(1 == len(state.shape))
        return self.cost(state[np.newaxis,:],action)[0]

class CostWrapper(CostFunction):
    def __init__(self,state_fn):
        self.state_fn = state_fn
        
    def cost(self,states,action):
        return self.state_fn.evaluate(states)
                

class DiscreteCostWrapper(CostFunction):
    def __init__(self,costs):
        self.costs = costs
    def cost(self,states,action):
        (N,d) = states.shape
        assert(1 == d)
        A = len(self.costs)
        assert(0 <= action < A)
        assert(0 == action % 1)
        
        assert(np.sum(np.fmod(states[:,0],1)) < 1e-15)
        state_ids = states[:,0].astype('i')
        return self.costs[action][state_ids]

class DiscreteMatchCost(CostFunction):
    def __init__(self,state):
        self.state = state
    def cost(self,states,action):
        (N,d) = states.shape
        assert(1==d)
        costs = np.ones(N)
        costs[states[:,0] == self.state] = 0
        return costs


class HillcarCost(CostFunction):
    def __init__(self):
        pass
        
    def cost(self,states,action):
        return np.linalg.norm(states,axis=1)

    def get_oob_costs(self):
        return self.oob_costs
