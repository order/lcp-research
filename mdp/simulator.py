import numpy as np
import pickle
import sys
import utils.batch as batch
import multiprocessing as mp

import linalg

class SimulationResults(object):
    def __init__(self,actions,
                 states,
                 costs,
                 internal_results=None):
        self.actions = actions
        self.states = states
        self.costs = costs
        self.internal_results = internal_results
        
    def merge(self,sim_results):
        """
        Results are n x T and m x T,
        Concatenate stacks them vertically
        """
        self.actions = np.concatenate([self.actions,
                                       sim_results.actions])
        self.states = np.concatenate([self.states,
                                       sim_results.states])
        self.costs = np.concatenate([self.costs,
                                      sim_results.costs])

        # Internal results is a list of T time steps, each of n
        if not self.internal_results\
           or not sim_results.internal_results:
            return
        
        T = len(self.internal_results)
        assert(T == len(sim_results.internal_results))
        for t in xrange(T):
            x = sim_results.internal_results[t]
            self.internal_results[t].extend(x)
        
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
    res = simulate(*args)
    return res
    
def simulate(problem,
             policy,
             start_states,
             rollout_horizon):

    H = rollout_horizon
    action_dim = policy.get_action_dim()
    (N,Sdim) = start_states.shape

    states = np.full((N,Sdim,H),np.nan,dtype=np.double)
    actions = np.full((N,action_dim,H),np.nan,dtype=np.double)
    costs = np.full((N,H),np.nan,dtype=np.double)

    
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
        
    return SimulationResults(actions,
                             states,
                             costs)

def discounted_return_with_tail_estimate(problem,
                                         costs,
                                         states,
                                         discount,
                                         ref_v_fn):
    H = costs.shape[-1]
    assert(H == states.shape[-1])
    final_states = states[:,:,-1]
    (N,d) = final_states.shape
    
    returns = discounted_return(costs,problem.discount)
    assert((N,) == returns.shape)
    
    tail_v = ref_v_fn.evaluate(final_states)
    assert((N,) == tail_v.shape)

    ret = returns + (discount**(H+1)) * tail_v
    assert((N,) == ret.shape)

    return ret
    


def discounted_return(cost,discount):
    (N,T) = cost.shape
    weight = np.power(discount,np.arange(T))
    R = cost.dot(weight)
    assert((N,) == R.shape)
    return R
