import numpy as np
import math
from mdp.bicycle import BicycleRemapper
from mdp.simulator import BicycleSimulator
from mdp.state_remapper import AngleWrapStateRemaper,RangeThreshStateRemapper
from mdp.node_mapper import InterpolatedGridNodeMapper
from mdp.discretizer import ContinuousMDPDiscretizer
from mdp.costs import QuadraticCost

import lcp.solvers

def simulate_test():
    """
    Simulates the bike to see the dynamics
    """
    physics = BicycleRemapper()
    S = len(physics.dim_ids)
    dim_ids = physics.dim_ids
    
    state = np.array(np.zeros((1,S)))
    state[0,dim_ids['omega']] = 0.01
    state[0,dim_ids['y']] = -physics.params.l
    state[0,dim_ids['psi']] = 0

    action = np.zeros(2)
    
    sim = BicycleSimulator(physics)
    sim.simulate(state,action,25)
  
def generate_discretizer(theta_desc,dtheta_desc,omega_desc,domega_desc,tau_desc,d_desc)
    cost_coef = np.ones(4)
    discount = 0.99
    
    raw_actions = util.grid_points(tau_desc,d_desc)
    single_actions = np.any(0 == raw_actions,axis=1)
    actions = raw_actions[single_actions,:]
    A = tau_desc[-1] + d_desc[-1] - 1
    assert((A,2) == actions.shape)
   
    physics = BicycleRemapper()
    basic_mapper = PiecewiseConstRegularGridNodeMapper(theta_desc,dtheta_desc,omega_desc,domega_desc)
    sink_node = basic_mapper.get_num_nodes() 

    # All oob -> sink node (Maybe threshold the velocities?)
    theta_oob = OOBSinkNodeMapper(0,-theta_lim,theta_lim,sink_node)
    dtheta_oob = OOBSinkNodeMapper(1,-dtheta_lim,dtheta_lim,sink_node)
    omega_oob = OOBSinkNodeMapper(2,-omega_lim,omega_lim,sink_node)
    domega_oob = OOBSinkNodeMapper(3,-domega_lim,domega_lim,sink_node)

    set_point = np.zeros(4)
    cost_type = 'target'
    if cost_type == 'quad':
        cost_obj = QuadraticCost(cost_coef,set_point)
    elif cost_type == 'ball':
        cost_obj = BallCost(set_point,0.1)
    elif cost_type == 'target':
        nudge = np.array([np.pi / 16.0,1.0,np.pi / 16.0,1.0])
        cost_obj = TargetZoneCost(np.array([set_point - nudge, set_point + nudge]))
    else:
        assert(False)
    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_node_mapper(theta_oob)
    discretizer.add_node_mapper(dtheta_oob)
    discretizer.add_node_mapper(omega_oob)
    discretizer.add_node_mapper(domega_oob)
    
if __name__ == '__main__':
    tN = 21
    vN = 21
    discretizer = generate_discretizer((-np.pi/2,np.pi/2,tN),\
        (-10,10,vN),\
        (-np.pi/16,np.pi/16,tN),\
        (-10,10,vN),\
        (-2,2,3),\
        (-2,2,3))
    simulate_only = False
    regen=True
    max_iter = 50000
    thresh = 1e-12
    discount = 0.95
    
    if simulate_only:
        policy = ConstantPolicy(np.zeros(2))
        init_state = np.array([[0.01,0.0,0.0,0.0]])
        plot_trajectory(discretizer,policy,init_state)
        quit()
        
    if regen:
        value_fun_eval = generate_value_function(discretizer,discount=discount,outfile='test_mdp.npz',max_iter=max_iter)
    else:
        value_fun_eval = generate_value_function(discretizer,discount=discount,filename='test_mdp.npz',outfile='test_mdp.npz',max_iter=max_iter,thresh=thresh)        

    plot_value_slice(discretizer,value_fun_eval,{2:0,3:0})
    K = 2
    policy = KStepLookaheadPolicy(discretizer, value_fun_eval, discount,K)
    #plot_policy_slice(discretizer,policy,{2:0,3:0})
    init_state = np.array([[0.01,0.0,0.0,0.0]])
    plot_trajectory(discretizer,policy,init_state)