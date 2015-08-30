from mdp.node_mapper import *
from mdp.state_remapper import *
from mdp.discretizer import *
from mdp.costs import *
from mdp.double_integrator import *

import random

import matplotlib.pyplot as plt
import time

import lcp.solvers

import numpy as np


def map_back_test():
    """
    Check to make sure that all nodes map back to themselves when run through the
    node -> state -> node machinery
    """
    
    np.set_printoptions(precision=3)
    
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

def plot_trajectory():
    np.set_printoptions(precision=3)
    
    n = basic_mapper.get_num_nodes()
    N = mdp_obj.costs[0].shape[0]

    nodes = [random.randint(0,n-1)]
    sink_nodes = [left_oob_mapper.sink_node, right_oob_mapper.sink_node]
    
    action_index = 0
    action = actions[action_index]
    for _ in xrange(5):
        new_nodes = set()
        for node in nodes:
            state = basic_mapper.nodes_to_states([node])
            elem = np.zeros((N,1))
            elem[node] = 1.0
            next_state_physics = physics.remap(state,action=action)
            node_dist = basic_mapper.states_to_node_dists(next_state_physics)
            print 'State {0} remaps to {1} via physics'.format(state.flatten(),next_state_physics.flatten())
            print 'Maps to:',node_dist[0]
            
            next_nodes = mdp_obj.transitions[action_index].dot(elem)
            for (j,val) in enumerate(next_nodes):
                if val > 0.05 and j not in sink_nodes:
                    next_state = basic_mapper.nodes_to_states([j])
                    print 'Drawing edge from {0}->{1} ({2}->{3}) '\
                        .format(node,j,state.flatten(),next_state.flatten())
                    plt.plot([state[0,0],next_state[0,0]],[state[0,1],next_state[0,1]],'.-')
                    new_nodes.add(j)
        nodes = new_nodes
    plt.show()    

def plot_value_function():
    MaxIter = 150

    vi = lcp.solvers.ValueIterator(mdp_obj)
    solver = lcp.solvers.IterativeSolver(vi)
    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    
    fn_eval = mdp.InterpolatedGridValueFunctionEvaluator(discretizer,vi.v)
    
    grid = 150
    [x_mesh,y_mesh] = np.meshgrid(np.linspace(-x_lim,x_lim,grid), np.linspace(-v_lim,v_lim,grid))
    Pts = np.array([x_mesh.flatten(),y_mesh.flatten()]).T
    
    vals = fn_eval.evaluate(Pts)
    

    Img = np.reshape(vals,(grid,grid))
    #Img = np.reshape(mdp_obj.costs[1][:basic_mapper.get_num_nodes()],(v_n,x_n))
    plt.imshow(Img,interpolation = 'nearest')
    plt.show()
    
def plot_interior_point():
    MaxIter = 50
    lcp_obj = mdp_obj.tolcp()
    print 'Built...', lcp_obj

    kip = lcp.solvers.KojimaIterator(lcp_obj)
    solver = lcp.solvers.IterativeSolver(kip)

    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    v = kip.get_primal_vector()[:basic_mapper.get_num_nodes()]
    Img = np.reshape(v[:basic_mapper.get_num_nodes()],(x_n,v_n)).T
    plt.imshow(Img,interpolation = 'nearest')
    plt.show()
    
def plot_projected_interior_point():
    kip = lcp.solvers.KojimaIterator(lcp_obj)
    solver = lcp.solvers.IterativeSolver(kip)

    solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
    print 'Starting solve...'
    solver.solve()
    print 'Done.'
    v = kip.get_primal_vector()[:basic_mapper.get_num_nodes()]
    Img = np.reshape(v[:basic_mapper.get_num_nodes()],(x_n,v_n)).T
    plt.imshow(Img,interpolation = 'nearest')
    plt.show()   
    
    
x_lim = 1
x_n = 20
xid = 0

v_lim = 3
v_n = 20
vid = 1

a_lim = 1
a_n = 3

OOBCost = 10

cost_coef = np.array([2,1])

basic_mapper = InterpolatedGridNodeMapper(np.linspace(-x_lim,x_lim,x_n),np.linspace(-v_lim,v_lim,v_n))
assert(basic_mapper.get_num_nodes() == x_n * v_n)
physics = DoubleIntegratorRemapper()
cost_obj = QuadraticCost(cost_coef)
actions = np.linspace(-a_lim,a_lim,a_n)

left_oob_mapper = OOBSinkNodeMapper(xid,-float('inf'),-x_lim,basic_mapper.num_nodes)
right_oob_mapper = OOBSinkNodeMapper(xid,x_lim,float('inf'),basic_mapper.num_nodes+1)
state_remapper = RangeThreshStateRemapper(vid,-v_lim,v_lim)

# Add oob as exceptions to the cost
cost_obj.override[left_oob_mapper.sink_node] = OOBCost
cost_obj.override[right_oob_mapper.sink_node] = OOBCost

discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
discretizer.add_state_remapper(state_remapper)
discretizer.add_node_mapper(left_oob_mapper)
discretizer.add_node_mapper(right_oob_mapper)

mdp_obj = discretizer.build_mdp()
print 'Built...', mdp_obj

#plot_interior_point()
#plot_trajectory()
plot_value_function()