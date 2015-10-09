from mdp.double_integrator import *
from mdp.costs import *
from mdp.discretizer import *
from mdp.node_mapper import *
from mdp.policy import *
from mdp.simulator import *
from mdp.state_remapper import *
from mdp.value_fun import *
import lcp.solvers

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import time


def map_back_test():
    """
    Check to make sure that all nodes map back to themselves when run through the
    node -> state -> node machinery
    """
    
    np.set_printoptions(precision=3)
    
    disc = generate_discretizer((-2,2,4),(-3,3,5),(0,1,2))
    basic_mapper = disc.basic_mapper
    
    mdp_obj = disc.build_mdp()
    
    n = basic_mapper.get_num_nodes()
    N = mdp_obj.costs[0].shape[0]
    for _ in xrange(125):
        node = random.randint(0,n-1)
        state = basic_mapper.nodes_to_states([node])
        back_again = basic_mapper.states_to_node_dists(state)[0]
        if back_again.keys()[0] != node:
            print 'Problem state',state
            print 'Node dist',back_again
            print '{0} != {1}'.format(node,back_again.keys()[0])
            assert(back_again.keys()[0] == node)
            
def plot_remap(discretizer,action,sink_states):
    states = discretizer.basic_mapper.get_node_states()
    next_states = discretizer.physics.remap(states,action=action)

    N = states.shape[0]
    for i in xrange(N):
        plt.plot([states[i,0],next_states[i,0]],[states[i,1],next_states[i,1]],'-b',lw=2)
    
    T = discretizer.states_to_transition_matrix(next_states)
    (Nodes,States) = T.nonzero()
    M = Nodes.size
    for i in xrange(M):
        node_id = Nodes[i]
        state_id = States[i]
        
        if node_id < N:
            x = states[node_id,0]
            y = states[node_id,1]
        else:
            x = sink_states[node_id - N,0]
            y = sink_states[node_id - N,1]    
        
        plt.plot([x,next_states[state_id,0]],[y,next_states[state_id,1]],'-r',alpha=T[node_id,state_id])
    plt.show()
    
def plot_costs(discretizer,action):
    (x_n,v_n) = discretizer.get_basic_len()
  
    states = discretizer.basic_mapper.get_node_states()
    costs = discretizer.cost_obj.cost(states,action)
    CostImg = np.reshape(costs,(x_n,v_n))
    plt.imshow(CostImg,interpolation = 'nearest')
    plt.title('Cost function')

    plt.show()       

def plot_trajectory(discretizer,policy):
    boundary = discretizer.get_basic_boundary()
    (x_lo,x_hi) = boundary[0]
    (v_lo,v_hi) = boundary[1]

    shade = 0.5
    x_rand = shade*random.uniform(x_lo,x_hi)
    v_rand = shade*random.uniform(v_lo,v_hi)
    init_state = np.array([[x_rand,v_rand]])
    assert((1,2) == init_state.shape)
    
    # Basic sanity
    discretizer.physics.remap(init_state,action=0) # Pre-animation sanity
    
    sim = DoubleIntegratorSimulator(discretizer)
    sim.simulate(init_state,policy,1000)    
    
    
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
    
    
def generate_discretizer(x_desc,v_desc,action_desc,**kwargs):
    xid,vid = 0,1 

    cost_coef = kwargs.get('cost_coef',np.ones(2))
    set_point = kwargs.get('set_point',np.zeros(2))
    oob_cost = kwargs.get('oob_costs',25.0)
    discount = kwargs.get('discount',0.99)
    assert(0 < discount < 1)

    basic_mapper = InterpolatedRegularGridNodeMapper(x_desc,v_desc)
    physics = DoubleIntegratorRemapper()
    #cost_obj = QuadraticCost(cost_coef,set_point,override=oob_cost)
    cost_obj = BallCost(np.array([0,0]),0.25)
    actions = np.linspace(*action_desc)

    (x_lo,x_hi,x_n) = x_desc
    (v_lo,v_hi,v_n) = v_desc
    
    left_oob_mapper = OOBSinkNodeMapper(xid,-float('inf'),x_lo,basic_mapper.num_nodes)
    right_oob_mapper = OOBSinkNodeMapper(xid,x_hi,float('inf'),basic_mapper.num_nodes+1)
    state_remapper = RangeThreshStateRemapper(vid,v_lo,v_hi)

    discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(state_remapper)
    discretizer.add_node_mapper(left_oob_mapper)
    discretizer.add_node_mapper(right_oob_mapper)
    
    mdp_obj = discretizer.build_mdp(discount=discount)

    return discretizer
    
def generate_value_function(discretizer,**kwargs):

    discount = kwargs.get('discount',0.99)
    max_iter = kwargs.get('max_iter',15)
    thresh = kwargs.get('thresh',1e-12)
    
    mdp_obj = discretizer.build_mdp(discount=discount)
    
    method = 'kojima'
    
    if method in ['kojima']:
        print 'Building LCP object...'
        start = time.time()
        lcp_obj = mdp_obj.tolcp()
        print 'Done. ({0:.2f}s)'.format(time.time() - start)        
    
    if method == 'value':
        iter = lcp.solvers.ValueIterator(mdp_obj)
    elif method == 'kojima':
        iter = lcp.solvers.KojimaIterator(lcp_obj)
    
    solver = lcp.solvers.IterativeSolver(iter)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(max_iter))
    if method in ['value']:
        solver.termination_conditions.append(lcp.util.ValueChangeTerminationCondition(thresh))
    elif method in ['kojima']:
        solver.termination_conditions.append(lcp.util.ResidualTerminationCondition(thresh))

    print 'Starting {0} solve...'.format(type(iter))
    start = time.time()
    solver.solve()
    print 'Done. ({0:.2f}s)'.format(time.time() - start)

    if method == 'value':
        J = iter.get_value_vector()
    elif method == 'kojima':
        N = mdp_obj.num_states
        J = iter.get_primal_vector()[:N]
        
    value_fun_eval = BasicValueFunctionEvaluator(discretizer,J)
    
    return value_fun_eval

if __name__ == '__main__':
    map_back_test() # Sanity test

    discount = 0.99
    discretizer = generate_discretizer((-4,4,100),(-6,6,100),(-1,1,3),cost_coef=np.array([1,0.5]))
    #plot_remap(discretizer,-1,np.array([[-6,-3],[6,3]]))

    value_fun_eval = generate_value_function(discretizer,discount=discount)
    #plot_costs(discretizer,-1)
    #plot_value_function(discretizer,value_fun_eval)
    #plot_advantage(discretizer,value_fun_eval,-1,1)
    #K = 1
    #policy = KStepLookaheadPolicy(discretizer, value_fun_eval, discount,K)
    #plot_policy(discretizer,policy)
    #plot_trajectory(discretizer,policy)