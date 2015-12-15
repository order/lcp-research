import numpy as np
import math
from mdp.pendulum import *
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

def plot_costs(discretizer,action):
    (x_n,v_n) = discretizer.get_basic_len()
  
    states = discretizer.basic_mapper.get_node_states()
    costs = discretizer.cost_obj.cost(states,action)
    CostImg = np.reshape(costs,(x_n,v_n))
    plt.imshow(CostImg,interpolation = 'nearest')
    plt.title('Cost function')

    plt.show()  
    
def plot_value_function(discretizer,value_fun_eval,**kwargs):
    boundary = discretizer.get_basic_boundary()    
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]
    
    grid = kwargs.get('grid_size',51)
    [x_mesh,v_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid), np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),v_mesh.flatten()])
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))
    
    three_d = False
    if three_d:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_mesh,v_mesh,ValueImg,rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    else:
        plt.pcolor(x_mesh,v_mesh,ValueImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Cost-to-go function')
    plt.show()   

def plot_policy(discretizer,policy,**kwargs):
    boundary = discretizer.get_basic_boundary()
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]
    
    grid = kwargs.get('grid_size',101)
    [x_mesh,y_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid), np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),y_mesh.flatten()])
    
    
    PolicyImg = np.reshape(policy.get_decisions(Pts),(grid,grid))

    plt.imshow(PolicyImg,interpolation = 'nearest')
    plt.title('Policy map')

    plt.show() 
    
def plot_advantage(discretizer,value_fun_eval,action1,action2):
    (x_n,v_n) = discretizer.get_basic_len()

    states = discretizer.basic_mapper.get_node_states()
    next_states1 = discretizer.physics.remap(states,action=action1)
    next_states2 = discretizer.physics.remap(states,action=action2)

    adv = value_fun_eval.evaluate(next_states1) - value_fun_eval.evaluate(next_states2)
    AdvImg = np.reshape(adv, (x_n,v_n))
    x_mesh = np.reshape(states[:,0], (x_n,v_n))
    v_mesh = np.reshape(states[:,1], (x_n,v_n))

    three_d = False
    if three_d:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x_mesh,v_mesh,AdvImg,rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    else:
        plt.pcolor(x_mesh,v_mesh,AdvImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Advantage function')
    plt.show()
    
def plot_trajectory(discretizer,policy,**kwargs):
    boundary = discretizer.get_basic_boundary()    
    q_rand = [random.uniform(x[0],x[1]) for x in boundary]

    frames = kwargs.get('frames',1000)

    init_state = kwargs.get('init_state',np.array([[0.01,0.0]]))
    assert((1,2) == init_state.shape)
    
    # Basic sanity
    discretizer.physics.remap(init_state,action=0) # Pre-animation sanity
    
    sim = PendulumSimulator(discretizer)
    sim.simulate(init_state,policy,frames)    
    
    
def generate_discretizer(theta_n,dtheta_desc,a_desc,**kwargs):
    theta_desc = (0.0,2*np.pi,theta_n)
    theta_grid = np.linspace(*theta_desc)    
    dtheta_grid = np.linspace(*dtheta_desc)    
    actions = np.linspace(*a_desc)
    
    cost_coef = np.ones(4)
    discount = 0.99
   
    physics = PendulumRemapper(length=2.0,dampening=0.5)
    basic_mapper = InterpolatedRegularGridNodeMapper(theta_desc,dtheta_desc)

    set_point = np.array([np.pi,0.0])
    cost_type = 'target'
    if cost_type == 'quad':
        cost_obj = QuadraticCost(cost_coef,set_point)
    elif cost_type == 'ball':
        cost_obj = BallCost(set_point,0.1,0.0,1.0)
    elif cost_type == 'target':
        nudge = np.pi/16.0
        cost_obj = TargetZoneCost(np.array([\
            [np.pi-nudge,np.pi+nudge],\
            [-np.pi/16.0,np.pi/16.0]]))
    else:
        assert(False)
    
    
    # Cylindrical boundaries
    q_angle_wrap = AngleWrapStateRemaper(0)
    dq_thres = RangeThreshStateRemapper(1,dtheta_desc[0],dtheta_desc[1])
    
    # Build discretizer
    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(q_angle_wrap)
    discretizer.add_state_remapper(dq_thres)
    
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
    discount = 0.999
    N = 100
    discretizer = generate_discretizer(N,(-10,10,N),(-3,3,3))

    simulate_only = False
    regen=True
    use_lqr = False

    max_iter = 50000
    thresh = 1e-9
    if not simulate_only or use_lqr:
        if regen:
            value_fun_eval = generate_value_function(discretizer,discount=discount,\
                outfile='test_mdp.npz',\
                max_iter=max_iter,thresh=thresh)
        else:
            value_fun_eval = generate_value_function(discretizer,discount=discount,\
                filename='test_mdp.npz',outfile='test_mdp.npz',\
                max_iter=max_iter,thresh=thresh)        

    plot_value_function(discretizer,value_fun_eval)
    plot_advantage(discretizer,value_fun_eval,-3,3)
    
    if use_lqr:
        (K,x) = generate_lqr(discretizer.physics,R = np.array([[1e-4]]))
        policy = LinearFeedbackPolicy(K,x,limit=(-3,3))
    else:
        K = 1
        policy = KStepLookaheadPolicy(discretizer, value_fun_eval, discount,K)

    plot_policy(discretizer,policy)
    x0 = np.array([[np.pi + 0.99*np.pi,0.0]])
    plot_trajectory(discretizer,policy,init_state = x0,frames = 2500)
