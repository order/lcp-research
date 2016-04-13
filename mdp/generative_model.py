import numpy as np

class GenerativeModel(object):
    def __init__(self,
                 trans_fn,
                 boundary,
                 cost_fn,
                 state_dim,
                 action_dim):

        self.trans_fn = trans_fn
        self.boundary = boundary
        self.cost_fns = cost_fn
        self.state_dim = state_dim
        self.action_dim = action_dim

    def multisample_next(self,states,action,S):
        (N,D) = states.shape
        assert(D == self.state_dim)
        assert((self.action_dim,) == action.shape
               or (N,self.action_dim) == action.shape)

        next_states = self.trans_fn.multisample_transition(states,
                                                           action,
                                                           S)
        assert((S,N,D) == next_states.shape)
        next_states = self.boundary.enforce(next_states)

        cost = self.cost_fns.cost(states,action)
        assert((N,) == cost.shape)
        return (next_states,cost)

    def next(self,states,action):
        (states,costs)=self.multisample_next(states,
                                             action,1)
        return (states[0,:,:],costs)
