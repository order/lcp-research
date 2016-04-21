import numpy as np
import pickle
import sys
import utils.batch as batch
import multiprocessing as mp

import linalg

class SimulationResults(object):
    def __init__(self,actions,
                 states,
                 costs):
        self.actions = actions
        self.states = states
        self.costs = costs
    def merge(self,sim_results):
        self.actions = np.concatenate([self.actions,
                                       sim_results.actions])
        self.states = np.concatenate([self.states,
                                       sim_results.states])
        self.costs = np.concatenate([self.costs,
                                      sim_results.costs])
        
    def shape(self):
        return (self.actions.shape,
                self.states.shape,
                self.costs.shape)

def batch_simulate(problem,
                   policy,
                   start_states,
                   rollout_horizon,
                   num_states_per_job,
                   workers=None):
    """
    Batch up the different start states into different
    """
    chunks = linalg.split(start_states,
                          num_states_per_job)
    num_jobs = len(chunks)
    if not workers:
        workers = min(mp.cpu_count()-1,num_jobs)

                                 
    print 'Starting {0} jobs on {1} workers'.format(num_jobs,
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

    states = np.empty((N,Sdim,H))
    actions = np.empty((N,action_dim,H))
    costs = np.empty((N,H))

    curr = start_states
    for t in xrange(H):
        u = policy.get_decisions(curr)
        assert((N,action_dim) == u.shape)
        (curr,cost) = problem.gen_model.multisample_next(curr,u,1)
        assert((1,N,Sdim) == curr.shape)
        assert((N,) == cost.shape)
        curr = curr[0,:,:]
        states[:,:,t] = curr
        actions[:,:,t] = u
        costs[:,t] = cost
    return SimulationResults(actions,states,costs)


def discounted_return(cost,discount):
    (N,T) = cost.shape
    weight = np.power(discount,np.arange(T))
    R = cost.dot(weight)
    assert((N,) == R.shape)
    return R
