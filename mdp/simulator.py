import numpy as np
import pickle
import sys
import utils.batch as batch
import multiprocessing as mp

def batch_simulate(problem,
                   policy,
                   start_states,
                   rollout_horizon,
                   num_states_per_job,
                   workers=None):
    """
    Batch up the different start states into different
    """
    if not workers:
        workers = mp.cpu_count()-1
    chunks = batch.break_ndarray(start_states,
                                 num_states_per_job)    
                                 
    print 'Starting {0} jobs on {1} workers'.format(len(chunks),
                                            workers)
    f = lambda S:simulate(problem,policy,S,rollout_horizon)
    res = batch.batch_process(f,chunks,workers)    

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
