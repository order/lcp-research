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

import time
import random


import lcp.solvers

def plot_value_slice(discretizer,value_fun_eval,fixed,**kwargs):

    assert(2 == len(fixed))
    boundary = discretizer.get_basic_boundary()    
    (theta_lo,theta_hi) = boundary[0]
    (dtheta_lo,dtheta_hi) = boundary[2]
    grid = kwargs.get('grid_size',125)
    
    t_lin = np.linspace(theta_lo,theta_hi,grid)
    dt_lin = np.linspace(dtheta_lo,dtheta_hi,grid)
    
    Lins = []
    for d in xrange(4):
        if d not in fixed:
            if d <= 1:
                Lins.append(t_lin)
            else:
                Lins.append(dt_lin)
    assert(2 == len(Lins))
    
    Meshes = np.meshgrid(*Lins ,indexing='ij')
    N = Meshes[0].size
    assert(N == grid*grid)
    
    Cols = []
    I = 0
    for d in xrange(4):
        if d not in fixed:
            Cols.append(Meshes[I].flatten())
            I += 1
        else:
            Cols.append(fixed[d]*np.ones(N))
    assert(4 == len(Cols))
    Pts = np.column_stack(Cols)
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))

    plt.imshow(ValueImg,interpolation = 'bicubic')
    plt.title('Cost-to-go function')

    plt.show() 
    
def plot_trajectory(discretizer,policy):
    boundary = discretizer.get_basic_boundary()    
    q_rand = [random.uniform(x[0],x[1]) for x in boundary]

    init_state = np.array([q_rand])
    init_state = np.array([[np.pi+0.005,0.0,0.0,0.0]])
    assert((1,4) == init_state.shape)
    
    # Basic sanity
    discretizer.physics.remap(init_state,action=0) # Pre-animation sanity
    
    sim = AcrobotSimulator(discretizer)
    sim.simulate(init_state,policy,1000)    
    
    
def generate_discretizer(theta_n,dtheta_desc,a_desc,**kwargs):
    theta_desc = (0.0,2*np.pi,theta_n)
    theta_grid = np.linspace(*theta_desc)    
    dtheta_grid = np.linspace(*dtheta_desc)    
    actions = np.linspace(*a_desc)
    
    cost_coef = np.ones(4)
    discount = 0.99
   
    physics = AcrobotRemapper()
    #basic_mapper = InterpolatedGridNodeMapper(theta_grid,theta_grid,dtheta_grid,dtheta_grid)
    #basic_mapper = PiecewiseConstRegularGridNodeMapper(theta_desc,theta_desc,dtheta_desc,dtheta_desc)
    basic_mapper = InterpolatedRegularGridNodeMapper(theta_desc,theta_desc,dtheta_desc,dtheta_desc)

    set_point = np.array([0.0,0.0,0.0,0.0])
    cost_obj = QuadraticCost(cost_coef,set_point)
    #cost_obj = BallCost(set_point,0.1,0.0,1.0)

    # Cylindrical boundaries
    q1_angle_wrap = AngleWrapStateRemaper(0)
    q2_angle_wrap = AngleWrapStateRemaper(1)
    dq1_thres = RangeThreshStateRemapper(2,dtheta_desc[0],dtheta_desc[1])
    dq2_thres = RangeThreshStateRemapper(3,dtheta_desc[0],dtheta_desc[1])
    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(q1_angle_wrap)
    discretizer.add_state_remapper(q2_angle_wrap)
    discretizer.add_state_remapper(dq1_thres)
    discretizer.add_state_remapper(dq2_thres)
    
    return discretizer
    
def generate_value_function(discretizer,**kwargs):
    discount = kwargs.get('discount',0.99)
    max_iter = kwargs.get('max_iter',500)
    thresh = kwargs.get('thresh',1e-12)
   
    # Build MDP
    print 'Starting MDP build...'
    start = time.time()
    mdp_obj = discretizer.build_mdp(discount=discount)
    print 'Done. ({0:.2f}s)'.format(time.time() - start)
    
    # Solve
    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(max_iter))
    solver.termination_conditions.append(lcp.util.ValueChangeTerminationCondition(thresh))
    print 'Starting solve...'
    start = time.time()
    solver.solve()
    print 'Done. ({0:.2f}s)'.format(time.time() - start)
    
    # Build policy
    value_fun_eval = BasicValueFunctionEvaluator(discretizer,vi.get_value_vector())
    return value_fun_eval
    
if __name__ == '__main__':
    discount = 1.0-1e-4
    discretizer = generate_discretizer(16,(-12,12,16),(-1,1,3))

    value_fun_eval = generate_value_function(discretizer,discount=discount)
    #plot_costs(discretizer,-1)
    plot_value_slice(discretizer,value_fun_eval,{2:1,3:0})
    #plot_advantage(discretizer,value_fun_eval,-1,1)
    K = 1
    #policy = KStepLookaheadPolicy(discretizer, value_fun_eval, discount,K)
    #plot_policy(discretizer,policy)
    #plot_trajectory(discretizer,policy)