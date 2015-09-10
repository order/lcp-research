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
    
def balance_mdp_test():
    """
    MDP for the balancing problem
    """
    
    # Acceptable ranges for 4 major parameters
    theta_lim = math.pi / 2.0
    theta_n = 10
    theta_desc = (-theta_lim,theta_lim,theta_n)
    
    dtheta_lim = 5
    dtheta_n = 10
    dtheta_desc = [-dtheta_lim,dtheta_lim,dtheta_n]
    
    omega_lim = math.pi / 15.0
    omega_n = 10
    omega_desc = (-omega_lim,omega_lim,omega_n)
    
    domega_lim = 0.5
    domega_n = 10
    domega_desc = (-domega_lim,domega_lim,domega_n)
    
    tau_lim = 2
    tau_n = 3
    tau_desc = (-tau_lim,tau_lim,tau_n)
    
    d_lim = 0.02
    d_n = 3
    d_desc = (-d_lim,d_lim,d_n)
    
    cost_coef = np.ones(4)
    discount = 0.99
   
    physics = BicycleRemapper()
    basic_mapper = PiecewiseConstRegularGridNodeMapper(theta_desc,dtheta_desc,omega_desc,domega_desc)
    cost_obj = QuadraticCost(cost_coef,np.zeros(4))
    sink_node = basic_mapper.get_num_nodes() 

    # All oob -> sink node (Maybe threshold the velocities?)
    theta_oob = OOBSinkNodeMapper(0,-theta_lim,theta_lim,sink_node)
    dtheta_oob = OOBSinkNodeMapper(1,-dtheta_lim,dtheta_lim,sink_node)
    omega_oob = OOBSinkNodeMapper(2,-omega_lim,omega_lim,sink_node)
    domega_oob = OOBSinkNodeMapper(3,-domega_lim,domega_lim,sink_node)

    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_node_mapper(theta_oob)
    discretizer.add_node_mapper(dtheta_oob)
    discretizer.add_node_mapper(omega_oob)
    discretizer.add_node_mapper(domega_oob)
    
    # Build MDP
    print 'Starting MDP build...'
    mdp_obj = discretizer.build_mdp(discount=discount)
    print 'Done.'
    
    # Solve
    MaxIter = 150
    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    solver.iter_message = '.'
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    
simulate_test()