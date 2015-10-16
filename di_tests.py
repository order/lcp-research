
import mdp
import mdp.double_integrator as di
import mdp.simulator as simulator

import solvers
from solvers.value_iter import ValueIterator
from solvers.kojima import KojimaIPIterator
from solvers.projective import ProjectiveIPIterator
from solvers.termination import *
from solvers.notification import *

import lcp

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


import time


def map_back_test():
    """
    Check to make sure that all nodes map back to themselves when 
    run through the
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

##########################################3
# Plotting functions
            
def plot_remap(discretizer,action,sink_states):
    states = discretizer.basic_mapper.get_node_states()
    next_states = discretizer.physics.remap(states,action=action)

    N = states.shape[0]
    for i in xrange(N):
        plt.plot([states[i,0],next_states[i,0]],\
                 [states[i,1],next_states[i,1]],'-b',lw=2)
    
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
        
        plt.plot([x,next_states[state_id,0]],\
                 [y,next_states[state_id,1]],'-r',alpha=T[node_id,state_id])
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
    [x_mesh,v_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid),\
                                  np.linspace(v_lo,v_hi,grid),indexing='ij')
    Pts = np.column_stack([x_mesh.flatten(),v_mesh.flatten()])
    
    vals = value_fun_eval.evaluate(Pts)
    ValueImg = np.reshape(vals,(grid,grid))
    
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
    [x_mesh,y_mesh] = np.meshgrid(np.linspace(x_lo,x_hi,grid),\
                                  np.linspace(v_lo,v_hi,grid),indexing='ij')
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

    adv = value_fun_eval.evaluate(next_states1)\
          - value_fun_eval.evaluate(next_states2)
    AdvImg = np.reshape(adv, (x_n,v_n))
    x_mesh = np.reshape(states[:,0], (x_n,v_n))
    v_mesh = np.reshape(states[:,1], (x_n,v_n))

    plt.pcolor(x_mesh,v_mesh,AdvImg)
        
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Advantage function')
    plt.show() 
    
#################################################
# Generate the DISCRETIZER object
    
def generate_discretizer(x_desc,v_desc,action_desc,**kwargs):
    xid,vid = 0,1 

    cost_coef = kwargs.get('cost_coef',np.ones(2))
    set_point = kwargs.get('set_point',np.zeros(2))
    oob_cost = kwargs.get('oob_costs',25.0)
    discount = kwargs.get('discount',0.99)
    assert(0 < discount < 1)

    basic_mapper = mdp.InterpolatedRegularGridNodeMapper(x_desc,v_desc)
    physics = di.DoubleIntegratorRemapper()
    #cost_obj = mdp.QuadraticCost(cost_coef,set_point,override=oob_cost)
    cost_obj = mdp.BallCost(np.array([0,0]),0.25)
    actions = np.linspace(*action_desc)

    (x_lo,x_hi,x_n) = x_desc
    (v_lo,v_hi,v_n) = v_desc
    
    left_oob_mapper = mdp.OOBSinkNodeMapper(xid,-float('inf'),x_lo,basic_mapper.num_nodes)
    right_oob_mapper = mdp.OOBSinkNodeMapper(xid,x_hi,float('inf'),basic_mapper.num_nodes+1)
    state_remapper = mdp.RangeThreshStateRemapper(vid,v_lo,v_hi)

    discretizer = mdp.ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
    discretizer.add_state_remapper(state_remapper)
    discretizer.add_node_mapper(left_oob_mapper)
    discretizer.add_node_mapper(right_oob_mapper)
    
    mdp_obj = discretizer.build_mdp(discount=discount)

    return discretizer

###########################################
# Build a PROJECTIVE LCP

def build_projective_lcp(mdp):
    

###########################################
# Find the VALUE FUNCTION
    
def find_value_function(discretizer,**kwargs):

    discount = kwargs.get('discount',0.99)
    max_iter = kwargs.get('max_iter',1000)
    thresh = kwargs.get('thresh',1e-6)
    method = kwargs.get('method','value')
    
    mdp_obj = discretizer.build_mdp(discount=discount)

    # Build the LCP object
    if method in ['kojima']:
        print 'Building LCP object...'
        start = time.time()
        lcp_obj = mdp_obj.tolcp()
        print 'Done. ({0:.2f}s)'.format(time.time() - start)

    # Build the Projective LCP object
    if method in ['projective']:
        print 'Building projected LCP object...'
        start = time.time()
        proj_lcp_obj = build_projective_lcp(mdp_obj)
        print 'Done. ({0:.2f}s)'.format(time.time() - start)        

    # Select the iterator
    if method == 'value':
        iter = ValueIterator(mdp_obj)
    elif method == 'kojima':
        iter = KojimaIPIterator(lcp_obj)
    else:
        raise NotImplmentedError()

    # Set up the solver object
    solver = solvers.IterativeSolver(iter)
    max_iter_cond = MaxIterTerminationCondition(max_iter)
    solver.termination_conditions.append(max_iter_cond)
    if method in ['value']:
        val_change_term = ValueChangeTerminationCondition(thresh)
        solver.termination_conditions.append(val_change_term)
        solver.notifications.append(ValueChangeAnnounce())        
    elif method in ['kojima']:
        res_change_term = ResidualTerminationCondition(thresh)
        solver.termination_conditions.append(res_change_term)
        solver.notifications.append(ResidualChangeAnnounce())


    # Actually do the solve
    print 'Starting {0} solve...'.format(type(iter))
    start = time.time()
    solver.solve()
    print 'Done. ({0:.2f}s)'.format(time.time() - start)

    # Extract the cost-to-go function
    if method == 'value':
        J = iter.get_value_vector()
    elif method == 'kojima':
        N = mdp_obj.num_states
        J = iter.get_primal_vector()[:N]

    # Place in an interpolating evaluator
    value_fun_eval = mdp.BasicValueFunctionEvaluator(discretizer,J)
    # TODO: use the low-dimension weights and basis if projective
    return value_fun_eval


#########################################
# Main function

if __name__ == '__main__':

    discount = 0.997
    max_iter = 1e3
    thresh = 1e-6

    x_desc = (-4,4,50)
    v_desc = (-6,6,50)
    a_desc = (-1,1,3)
    cost_coef = np.array([1,0.5])

    discretizer = generate_discretizer(x_desc,v_desc,a_desc,\
                                       cost_coef=cost_coef)
    #plot_remap(discretizer,-1,np.array([[-6,-3],[6,3]]))

    value_fun_eval = generate_value_function(discretizer,\
                                             discount=discount,\
                                             max_iter=max_iter,\
                                             thresh=thresh)
    #plot_costs(discretizer,-1)
    #plot_value_function(discretizer,value_fun_eval)
    #plot_advantage(discretizer,value_fun_eval,-1,1)
    lookahead = 3
    policy = mdp.KStepLookaheadPolicy(discretizer, value_fun_eval, discount,lookahead)
    plot_policy(discretizer,policy)
    #plot_trajectory(discretizer,policy)
