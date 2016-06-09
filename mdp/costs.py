import state_functions
import numpy as np

class CostFunction(object):
    def cost(self,states,action):
        raise NotImplementedError()
    def single_cost(self,state,action):
        assert(1 == len(state.shape))
        return self.cost(state[np.newaxis,:],action)[0]
    def get_oob_costs(self):
        raise NotImplementedError()

class CostWrapper(CostFunction):
    def __init__(self,state_fn):
        self.favored = np.empty(0)
        self.state_fn = state_fn
        self.nudge = 1e-4
        
    def cost(self,states,action):
        c = self.state_fn.evaluate(states)
        if self.favored.size == 0:
            return c

        if (2 == len(action.shape)):
            mask = np.any(action != self.favored,axis=1)
        else:
            assert(1 == len(action.shape))
            mask = np.any(action != self.favored)
        return c + self.nudge * mask

    def get_oob_costs(self):
        return self.oob_costs
                

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
