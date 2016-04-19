import numpy as np
import pickle
import sys

def simulate(problem,
             policy,
             start_states,
             rollout_horizon):

    H = rollout_horizon
    action_dim = policy.get_action_dim()
    (N,Sdim) = start_states.shape

    states = np.empty((H,N,Sdim))
    actions = np.empty((H,N,action_dim))
    costs = np.empty((H,N))

    curr = start_states
    for t in xrange(H):
        u = policy.get_decisions(curr)
        assert((N,action_dim) == u.shape)
        (curr,cost) = problem.gen_model.multisample_next(curr,u,1)
        assert((1,N,Sdim) == curr.shape)
        assert((N,) == cost.shape)
        curr = curr[0,:,:]
        states[t,:,:] = curr
        actions[t,:,:] = u
        costs[t,:] = cost

    return (actions,states,costs)


def discounted_return(cost,discount):
    (T,N) = cost.shape
    weight = np.power(discount,np.arange(T))
    R = weight.dot(cost)
    assert((N,) == R.shape)
    return R
