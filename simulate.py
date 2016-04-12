import numpy as np
import pickle
import sys

def simulate(gen_model,
             discount,
             policy,
             start_states,
             rollout_horizon):

    H = rollout_horizon
    Adim = action_dim
    (N,Sdim) = start_states.shape

    states = np.empty(H,N,Sdim)
    actions = np.empty(H,N,Adim)
    cost = np.empty(H,N)

    curr = start_states
    for t in xrange(H):
        u = self.get_decisions(curr)
        assert((N,Adim) == u.shape)
        (curr,cost) = gen_model.next(curr,u)

        states[t,:,:] = curr
        actions[t,:,:] = u
        costs[t,:] = cost

    return (actions,states,costs)

if __name__ == '__main__':
    (_,gen_pickle,policy_pickle,discount,horizon) = sys.argv

    FH = open(gen_pickle,'w')
    gen_model = pickle.load(FH)

    FH = open(policy_pickle,'w')
    policy = pickle.load(FH)

    start_states = gen_model.boundary.random_points(15)

    (actions,states,costs) = simulate(gen_model,
                                      discount,
                                      policy,
                                      start_states
                                      horizon)
    
