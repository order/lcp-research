import pickle

import numpy as np
import matplotlib.pyplot as plt

from config.instance.double_integrator\
    import DoubleIntegratorConfig as inst_conf

from mdp.policy import ConstantPolicy, MinFunPolicy, MaxFunPolicy
from mdp.state_functions import InterpolatedFunction
from mdp.q_estimation import get_q_vectors

import utils
from utils.plotting import cdf_points

class SimulationObject(object):
    def __init__(self,problem,policy):
        self.policy = policy
        self.problem = problem
        self.cost_obj = problem.cost_obj

    def next(self,x):
        (N,d) = x.shape
        assert(not np.any(np.isnan(x)))

        # Get the actions
        #actions = self.policy.get_decisions(x)
        actions = (-0.05*x[:,0])[:,np.newaxis]
        
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

    policy = ConstantPolicy(0 * np.ones(1))
    sim_obj = SimulationObject(problem,policy)
    x = uniform(R,boundaries)
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

###############
# Entry point #
###############
if __name__ == '__main__':
    data = np.load('data/test.npz')
    params = pickle.load(open('data/test.pickle','rb'))
    
    discretizer = params['instance_builder']
    problem = discretizer.problem
    mdp_obj = params['objects']['mdp']

    N = discretizer.get_num_nodes()
    A = discretizer.num_actions
    B = discretizer.problem.boundary
    
    G = 101  
    lins = ([np.linspace(l,h,G) for (l,h) in B])
    P = utils.make_points(*lins)
    assert((G*G,2) == P.shape)

    # V
    v = data['primal'][-1,:N]
    v_fn = InterpolatedFunction(discretizer,v)
    V = v_fn.evaluate(P)

    #Q
    q = get_q_vectors(mdp_obj,v)
    Q = np.empty((G*G,A))
    q_fns = []
    for a in xrange(A):
        q_fn = InterpolatedFunction(discretizer,q[:,a])
        q_fns.append(q_fn)
        Q[:,a] = q_fn.evaluate(P)

    #F
    f = np.reshape(data['primal'][-1,N:],(N,A),order='F')
    F = np.empty((G*G,A))
    f_fns = []
    for a in xrange(A):
        f_fn = InterpolatedFunction(discretizer,f[:,a])
        f_fns.append(f_fn)
        F[:,a] = f_fn.evaluate(P)
        
    SortedFlow = np.sort(F,axis=1)
    FlowAdv = SortedFlow[:,-1] - SortedFlow[:,-2]
    
    Policy = np.argmin(Q,axis=1)
    PolicyFlow = np.argmax(F,axis=1)

    R = 100
    I = 1000
    q_policy = MinFunPolicy(mdp_obj.actions,
                            q_fns)
    flow_policy = MaxFunPolicy(mdp_obj.actions,
                               f_fns)
    q_traces = simulate(problem,q_policy,R,I)
    flow_traces = simulate(problem,q_policy,R,I)

    #print q_traces[0].shape
    #quit()

    if False:
        f, axarr = plt.subplots(3,2)
        axarr[0][0].pcolor(np.reshape(V,(G,G)))
        axarr[1][0].pcolor(np.reshape(Policy,(G,G)))
        axarr[2][0].plot(q_traces[0][:,:,0].T,
                         q_traces[0][:,:,1].T,
                         '-b',
                         alpha=0.25)
    
        axarr[0][1].pcolor(np.reshape(FlowAdv,(G,G)))
        axarr[1][1].pcolor(np.reshape(PolicyFlow,(G,G)))
        axarr[2][1].plot(flow_traces[0][:,:,0].T,
                         flow_traces[0][:,:,1].T,
                         '-r',
                         alpha=0.25)

    gamma = np.power(mdp_obj.discount,np.arange(I))
    q_returns = np.sum(q_traces[1] * gamma,axis=1)
    assert((R,) == q_returns.shape)
    
    f_returns = np.sum(flow_traces[1] * gamma,axis=1)
    assert((R,) == f_returns.shape)
    
    (qx,qy) = cdf_points(q_returns)
    (fx,fy) = cdf_points(f_returns)
    plt.plot(qx,qy,'-b',fx,fy,'-r')

    plt.show()
    

    
