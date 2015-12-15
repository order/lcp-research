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
import os

import lcp.solvers

def generate_slice(discretizer,fixed,**kwargs):
    assert(2 == len(fixed))
    boundary = discretizer.get_basic_boundary()    
    grid = kwargs.get('grid_size',125)
    
    Lins = [np.linspace(lo,hi,grid) for (lo,hi) in boundary]
    Lins = [Lins[d] for d in xrange(4) if d not in fixed]
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
    return np.column_stack(Cols)
    
def plot_value_slice(discretizer,value_fun_eval,fixed,**kwargs):
    grid = kwargs.get('grid_size',125)
    Pts = generate_slice(discretizer,fixed,**kwargs)
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))
    plt.imshow(ValueImg,interpolation = 'bicubic')
    plt.title('Cost-to-go function')
    plt.show() 

def plot_policy_slice(discretizer,policy,fixed,**kwargs):

    grid = kwargs.get('grid_size',125)
    Pts = generate_slice(discretizer,fixed,**kwargs)
  
    dec = policy.get_decisions(Pts)
    DecImg = np.reshape(dec,(grid,grid))
    plt.imshow(DecImg,interpolation = 'bicubic')
    plt.title('Cost-to-go function')
    plt.show() 
    
def plot_trajectory(discretizer,policy,init_state):
    boundary = discretizer.get_basic_boundary()    
    q_rand = [random.uniform(x[0],x[1]) for x in boundary]
    assert((1,4) == init_state.shape)
    
    # Basic sanity
    discretizer.physics.remap(init_state,action=0) # Pre-animation sanity
    
    sim = AcrobotSimulator(discretizer)
    sim.simulate(init_state,policy,2500)    
    
    
def generate_discretizer(theta1_desc,theta2_desc,dtheta1_desc,dtheta2_desc,a_desc,**kwargs):
    actions = np.linspace(*a_desc)
    
    cost_coef = np.ones(4)
    discount = kwargs.get('discount',0.93)
   
    physics = AcrobotRemapper(l2=1.0)
    basic_mapper = InterpolatedRegularGridNodeMapper(theta1_desc,theta2_desc,dtheta1_desc,dtheta2_desc)

    set_point = np.array([np.pi,0.0,0.0,0.0])
    #set_point = np.array([0.0,0.0,0.0,0.0])
    cost_type = 'target'
    if cost_type == 'quad':
        cost_obj = QuadraticCost(cost_coef,set_point)
    elif cost_type == 'ball':
        cost_obj = BallCost(set_point,0.1)
    elif cost_type == 'target':
        nudge = np.array([np.pi / 16.0,np.pi / 16.0,1.0,1.0])
        cost_obj = TargetZoneCost(np.array([set_point - nudge, set_point + nudge]))
    else:
        assert(False)
    
    
    # Cylindrical boundaries
    q1_angle_wrap = WrapperStateRemaper(0,theta1_desc[0],theta1_desc[1])
    q2_angle_wrap = WrapperStateRemaper(1,theta2_desc[0],theta2_desc[1])
    dq1_thres = RangeThreshStateRemapper(2,dtheta1_desc[0],dtheta1_desc[1])
    dq2_thres = RangeThreshStateRemapper(3,dtheta2_desc[0],dtheta2_desc[1])
    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(q1_angle_wrap)
    discretizer.add_state_remapper(q2_angle_wrap)
    discretizer.add_state_remapper(dq1_thres)
    discretizer.add_state_remapper(dq2_thres)
    
    return discretizer
    
def generate_value_function(discretizer,**kwargs):
    discount = kwargs.get('discount',0.99)
    max_iter = kwargs.get('max_iter',100)
    thresh = kwargs.get('thresh',1e-8)
   
    # Build MDP
    if 'filename' in kwargs and os.path.isfile(kwargs['filename']):
        print 'Starting MDP load...'
        start = time.time()
        mdp_obj = mdp.MDP(kwargs['filename'])
        print 'Done. ({0:.2f}s)'.format(time.time() - start)

    else:
        print 'Starting MDP build...'
        start = time.time()
        mdp_obj = discretizer.build_mdp(discount=discount)
        print 'Done. ({0:.2f}s)'.format(time.time() - start)
    if 'outfile' in kwargs:
        mdp_obj.write(kwargs['outfile'])
        
    # Solve
    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(max_iter))
    solver.termination_conditions.append(lcp.util.ValueChangeTerminationCondition(thresh))
    solver.notifications.append(lcp.util.ValueChangeAnnounce())
    
    print 'Starting solve...'
    start = time.time()
    solver.solve()
    print 'Done. ({0:.2f}s)'.format(time.time() - start)
    
    # Build policy
    value_fun_eval = BasicValueFunctionEvaluator(discretizer,vi.get_value_vector())
    return value_fun_eval
    
if __name__ == '__main__':
    tN = 18
    vN = 18
    theta1 = (0.0,2.0*np.pi,tN)
    theta2 = (-np.pi,np.pi,tN)
    velo = (-10,10,vN)
    actions = (-10,10,5)
    discretizer = generate_discretizer(theta1,theta2,velo,velo,actions)
    
    simulate_only = False
    regen=True
    max_iter = 50000
    thresh = 1e-12
    discount = 0.95
    
    if simulate_only:
        policy = ConstantPolicy(0)        
        init_state = np.array([[np.pi+0.1,0.0,0.0,0.0]])
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
    init_state = np.array([[0.0,0.0,0.0,0.0]])
    plot_trajectory(discretizer,policy,init_state)
