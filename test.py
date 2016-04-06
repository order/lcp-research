import numpy as np
import matplotlib.pyplot as plt

import time

import discrete.regular_interpolate as reg_interp
from discrete.discretize import make_points

import mdp.transition_functions.double_integrator as di
import mdp.transition_functions.mdp_wrapper as mdp_wrapper
import mdp.state_remapper as remapper
import mdp.state_functions as state_fn
import mdp.costs as costs
import mdp.mdp_builder as mdp_builder
import solvers
from solvers.value_iter import ValueIterator
import utils

import mcts

# Transition function
di_params = utils.kwargify(step=0.01,
                           num_steps=5,
                           dampening=0.01,
                           control_jitter=0)
transition_function = di.DoubleIntegratorTransitionFunction(**di_params)

# Discretizer
X = 74
Y = 74
x_grid_desc = (-5,5,X)
v_grid_desc = (-5,5,Y)
grid_descs = [x_grid_desc,v_grid_desc]
discretizer = reg_interp.RegularGridInterpolator(grid_descs)


# State_remappers
state_remappers = [remapper.RangeThreshStateRemapper(0,-5,5),
                  remapper.RangeThreshStateRemapper(1,-5,5)]

# Costs
state_fun = state_fn.BallSetFn(np.zeros(2), 0.5)
#state_fun = state_fn.GaussianFn(1,np.zeros(2),9)
cost_function = costs.CostWrapper(state_fun)

# Actions:
actions = np.array([[-1],[0],[1]])
disc_actions = np.array([[0],[1],[2]])

# Discount
discount = 0.97

# Samples
samples = 5

builder = mdp_builder.MDPBuilder(transition_function,
                                 discretizer,
                                 state_remappers,
                                 cost_function,
                                 actions,
                                 discount,
                                 samples,
                                 False)
mdp_obj = builder.build_mdp()


disc_trans = mdp_wrapper.MDPTransitionWrapper(
    mdp_obj.transitions)
disc_cost = costs.DiscreteCostWrapper(mdp_obj.costs)
points = discretizer.get_cutpoints()
N = mdp_obj.num_states

target = np.array([-1,1])
dist = np.sum(np.abs(points[:-1,:] - target),axis=1)
start_id = np.argmin(dist)
#start_id =np.random.randint(N) 
start_state = np.array([start_id])
start_pos = points[start_id,:]
print 'Start index', start_id
print 'Start state', start_pos

tree = mcts.MonteCarloTree(disc_trans,
                           disc_cost,
                           discount,
                           disc_actions,
                           start_state,100)
for i in xrange(1000):
    print '-'*15
    (path,a_list) = tree.path_to_leaf()
    leaf = path[-1]

    # Expand child
    child_node = tree.expand_leaf(leaf)
    
    (G,a_id,state,cost) = tree.rollout(child_node.state)
    

    a_list.append(a_id)
    assert(len(a_list) == len(path))
    tree.backup(path,a_list,G)
    print i,tree.root_node.value
    
#mcts.display_tree(tree.root_node)

