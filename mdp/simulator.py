import numpy as np
import pickle
import sys
import utils.batch as batch
import multiprocessing as mp

class SimulationResults(object):
    def __init__(self,actions,
                 states,
                 costs):
        self.actions = actions
        self.states = states
        self.costs = costs
    def merge(self,sim_results):
        self.actions = np.vstack([self.actions,
                                  sim_results.actions])
        self.states = np.vstack([self.states,
                                  sim_results.states])
        self.costs = np.vstack([self.costs,
                                  sim_results.costs])    

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
    args = [(problem,policy,x,rollout_horizon) for x in chunks]
    res = batch.batch_process(dummy_simulate,args,workers)

    comb_results = res[0]
    for i in xrange(1,len(res)):
        comb_results.merge(res[i])
    return comb_results

def dummy_simulate(args):
    return simulate(*args)
    
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

    return SimulationResults(actions,states,costs)


def discounted_return(cost,discount):
    (T,N) = cost.shape
    weight = np.power(discount,np.arange(T))
    R = weight.dot(cost)
    assert((N,) == R.shape)
    return R
