import numpy as np
import math
from mdp.acrobot import *
from mdp.costs import *
from mdp.discretizer import *
from mdp.node_mapper import *
from mdp.policy import *
from mdp.simulator import *
from mdp.state_remapper import *
from mdp.value_fun import *


import lcp.solvers

def simulate_test():
    physics = AcrobotRemapper()
    init_state = np.array([math.pi/4.0,0.0,0,0])

    physics.remap(init_state,action=0)
    physics.forward_kinematics(init_state)
    policy = ConstantPolicy(1)

    sim = AcrobotSimulator(physics)
    sim.simulate(init_state,policy,10)
    
def mdp_test():
    theta_n = 25
    theta_desc = [0,2*math.pi,theta_n+1]
    theta_grid = np.linspace(*theta_desc)
    
    dtheta_n = 25
    dtheta_lim = 10
    dtheta_desc = [-dtheta_lim,dtheta_lim,dtheta_n]
    dtheta_grid = np.linspace(*dtheta_desc)
    
    a_lim = 1
    a_n = 3
    actions = np.linspace(-a_lim,a_lim,a_n)
    
    cost_coef = np.ones(4)
    discount = 0.99
   
    physics = AcrobotRemapper()
    #basic_mapper = InterpolatedGridNodeMapper(theta_grid,theta_grid,dtheta_grid,dtheta_grid)
    #basic_mapper = PiecewiseConstRegularGridNodeMapper(theta_desc,theta_desc,dtheta_desc,dtheta_desc)
    basic_mapper = InterpolatedRegularGridNodeMapper(theta_desc,theta_desc,dtheta_desc,dtheta_desc)

    cost_obj = QuadraticCost(cost_coef,np.array([math.pi,0.0,0.0,0.0]))

    # Cylindrical boundaries
    q1_angle_wrap = AngleWrapStateRemaper(0)
    q2_angle_wrap = AngleWrapStateRemaper(1)
    dq1_thres = RangeThreshStateRemapper(2,-dtheta_lim,dtheta_lim)
    dq2_thres = RangeThreshStateRemapper(3,-dtheta_lim,dtheta_lim)
    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(q1_angle_wrap)
    discretizer.add_state_remapper(q2_angle_wrap)
    discretizer.add_state_remapper(dq1_thres)
    discretizer.add_state_remapper(dq2_thres)
    
    # Build MDP
    print 'Starting MDP build...'
    mdp_obj = discretizer.build_mdp(discount=discount)
    print 'Done.'
    
    # Solve
    MaxIter = 2500
    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    solver.termination_conditions.append(lcp.util.ValueChangeTerminationCondition(1e-12))
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    
    # Build policy
    value_fun_eval = BasicValueFunctionEvaluator(discretizer,vi.get_value_vector())
    policy = OneStepLookaheadPolicy(cost_obj, physics, value_fun_eval, actions,discount)
    #policy = ConstantPolicy(1)
    
    # Simulate
    init_state = np.array([math.pi/4.0,0.0,0,0])
    physics.remap(init_state,action=0) # Pre-animation sanity
    physics.forward_kinematics(init_state)
    
    sim = AcrobotSimulator(physics)
    sim.simulate(init_state,policy,1000)
    
mdp_test()