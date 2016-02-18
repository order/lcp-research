import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from config.instance.double_integrator\
    import DoubleIntegratorConfig as inst_conf

from mdp.policy import ConstantPolicy, MinFunPolicy, MaxFunPolicy
from mdp.state_functions import InterpolatedFunction
from mdp.q_estimation import get_q_vectors

import utils
from utils.plotting import cdf_points

import config

class SimulationObject(object):
    def __init__(self,problem,policy):
        self.policy = policy
        self.problem = problem
        self.cost_obj = problem.cost_obj

    def next(self,x):
        (N,d) = x.shape
        assert(not np.any(np.isnan(x)))

        # Get the actions
        actions = self.policy.get_decisions(x)
        #actions = (-0.05*x[:,0])[:,np.newaxis]
        
        # Run the physics
        x_next = self.problem.next_states(x,actions)
        assert(x.shape == x_next.shape)
        assert(not np.any(np.isnan(x)))
        # NAN-out unfixed oobs
        
        oobs = self.problem.out_of_bounds(x_next)
        x_next[oobs,:] = np.nan
        
        costs = self.cost_obj.evaluate(x_next) # Should have NAN handling
        
        return (x_next,costs,actions,oobs)       

def uniform(N,boundaries):
    D = len(boundaries)
    x = np.empty((N,D))
    for (i,(low,high)) in enumerate(boundaries):        
        x[:,i] = np.random.uniform(low,high,N)
    return x 

def simulate(problem,policy,R,I):
    D = problem.get_dimension()
    aD = problem.action_dim
    boundaries = problem.get_boundary()

    oob_cost = problem.cost_obj.evaluate(np.full((1,D),np.nan))
    costs = np.full((R,I),oob_cost) # Default to oob_cost
    states = np.full((R,I,D),np.nan) # Default to nan.
    actions = np.full((R,I,aD),np.nan) 

    sim_obj = SimulationObject(problem,policy)
    x = uniform(R,boundaries)
    #x = np.tile(np.array([[1,0]]),(R,1))
    in_bounds = np.ones(R,dtype=bool)
    for i in xrange(I):
        (x,c,a,new_oob) = sim_obj.next(x)
        assert(not np.any(np.isnan(c)))

        assert(in_bounds.sum() == x.shape[0])
        states[in_bounds,i,:] = x
        costs[in_bounds,i] = c
        actions[in_bounds,i,:] = a

        # Crop out terminated sequences
 
        x = x[~new_oob,:]        
        in_bounds[in_bounds] = ~new_oob # Update what is `inbounds'

    return (states,costs,actions)

def get_policy(policy_file,data,params):
    policy_gen = utils.get_instance_from_file(policy_file)    
    assert(issubclass(type(policy_gen),
                      config.PolicyGenerator))
    return policy_gen.generate_policy(data,params)


###############
# Entry point #
###############
if __name__ == '__main__':

    if 6 != len(sys.argv):
        print 'Usage: <data file> <policy file> <#runs> <#iters> <save file>'
        quit()
    (_,data_file,policy_file,runs,iters,save_file) = sys.argv

    assert(data_file.endswith('.npz'))
    root_file = data_file[:-3]
    pickle_file = root_file + 'pickle'
    data = np.load(data_file)
    params = pickle.load(open(pickle_file,'rb'))

    R = int(runs)
    I = int(iters)
    
    problem = params['instance_builder'].problem
    mdp_obj = params['objects']['mdp']

    # BUILD POLICY
    policy = get_policy(policy_file, data, params)

    # Simulate
    (states,costs,actions) = simulate(problem,policy,R,I)

    (r,i,d) = states.shape
    assert((R,I) == (r,i))
    (r,i,ad) = actions.shape
    assert((R,I) == (r,i))

    if d == 2 and ad == 1:
        plt.plot(states[:,:,0].T,
                 states[:,:,1].T,
                 '-b',
                 alpha=0.5)
        plt.scatter(states[:,:,0].flatten(),
                    states[:,:,1].flatten(),
                    c = actions.flatten(),
                    edgecolors='none')
        plt.show()    

    # Find the returns
    gamma = np.power(mdp_obj.discount,np.arange(I))
    returns = np.sum(costs * gamma,axis=1)
    assert((R,) == returns.shape)

    np.save(save_file,returns)

    
