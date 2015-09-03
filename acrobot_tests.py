import numpy as np
import math
from mdp.acrobot import AcrobotRemapper
from mdp.simulator import ChainSimulator
from mdp.state_remapper import AngleWrapStateRemaper,RangeThreshStateRemapper
from mdp.node_mapper import InterpolatedGridNodeMapper
from mdp.discretizer import ContinuousMDPDiscretizer
from mdp.costs import QuadraticCost

def simulate_test():
    physics = AcrobotRemapper(l1=2)
    init_state = np.array([[math.pi/4.0,0.0,0,0]])

    physics.remap(init_state,action=0)
    physics.forward_kinematics(init_state)

    dim = 2
    sim = ChainSimulator(dim,physics)
    sim.simulate(init_state,1000)
    
def mdp_test():
    theta_n = 15
    theta_grid = np.linspace(0,2*math.pi,theta_n+1)[:-1]
    
    dtheta_n = 12
    dtheta_lim = 10
    dtheta_grid = np.linspace(-dtheta_lim,dtheta_lim,dtheta_n)
    
    a_lim = 1
    a_n = 3
    actions = np.linspace(-a_lim,a_lim,a_n)
    
    cost_coef = np.array([1,1])
    discount = 0.99
   
    physics = AcrobotRemapper()
    basic_mapper = InterpolatedGridNodeMapper(theta_grid,theta_grid,dtheta_grid,dtheta_grid)
    cost_obj = QuadraticCost(cost_coef,setpoint=np.array([math.pi,0.0,0.0,0.0]))

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
    mdp_obj = discretizer.build_mdp(discount=discount)

    # Solve
    MaxIter = 150
    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    solver.iter_message = '.'
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    
mdp_test()