import numpy as np
import matplotlib.pyplot as plt

import time

import discrete.regular_interpolate as reg_interp
import discrete.discretize as discretize

import mdp.transition_functions.double_integrator as di
import mdp.transition_functions.mdp_wrapper as mdp_wrapper

import mdp.policy
import mdp.state_remapper as remapper
import mdp.state_functions as state_fn
import mdp.costs as costs
import mdp.mdp_builder as mdp_builder

import solvers
from solvers.kojima import KojimaIPIterator
from solvers.value_iter import ValueIterator

import utils

import mcts

def make_di_mdp():

    trans_params = utils.kwargify(step=0.01,
                                  num_steps=5,
                                  dampening=0.01,
                                  control_jitter=0.01)
    trans_fn = di.DoubleIntegratorTransitionFunction(
        **trans_params)
    discretizer = reg_interp.RegularGridInterpolator([(-5,5,40),
                                                      (-5,5,40)])
    x_bound = remapper.RangeThreshStateRemapper(0,-5,5)
    v_bound = remapper.RangeThreshStateRemapper(1,-5,5)
    state_remappers = [x_bound,v_bound]

    cost_state_fn = state_fn.BallSetFn(np.zeros(2), 0.5)
    cost_fn = costs.CostWrapper(cost_state_fn)

    actions = np.array([[-1,0,1]]).T

    discount = 0.997

    num_samples = 10

    builder = mdp_builder.MDPBuilder(trans_fn,
                                     discretizer,
                                     state_remappers,
                                     cost_fn,
                                     actions,
                                     discount,
                                     num_samples,
                                     False)
    mdp_obj = builder.build_mdp()

    return builder,mdp_obj

def solve_mdp_kojima(mdp_obj):
    lcp_obj = mdp_obj.build_lcp()
    iterator = KojimaIPIterator(lcp_obj,
                                val_reg=0.0,
                                flow_reg=1e-12)
    
    it_solver = solvers.IterativeSolver(iterator)
    it_solver.termination_conditions.append(
        solvers.PrimalChangeTerminationCondition(1e-12))
    it_solver.termination_conditions.append(
        solvers.MaxIterTerminationCondition(1e3))
    it_solver.notifications.append(
        solvers.PrimalChangeAnnounce())

    it_solver.solve()
    return iterator.get_primal_vector()

def solve_mdp_value(mdp_obj):
    iterator = ValueIterator(mdp_obj)
    
    it_solver = solvers.IterativeSolver(iterator)
    it_solver.termination_conditions.append(
        solvers.ValueChangeTerminationCondition(1e-6))
    it_solver.termination_conditions.append(
        solvers.MaxIterTerminationCondition(1e4))
    it_solver.notifications.append(
        solvers.ValueChangeAnnounce())

    it_solver.solve()
    return iterator.get_value_vector()

def make_interps(discretizer,x):
    (N,) = x.shape
    n = discretizer.num_nodes
    A = (N / n) - 1
    assert((A % 1) < 1e-15)
    A = int(A)
    v = x[:n]

    V = state_fn.InterpolatedFunction(discretizer,x[:n])
    #V = state_fn.BallSetFn(np.zeros(2), 0.25)
    
    Qs = []
    for a in xrange(A):
        idx = slice((a+1)*n,(a+2)*n)
        q = state_fn.InterpolatedFunction(discretizer,x[idx])
        Qs.append(q)
    return V,Qs
    
def make_tree(builder,value_fn,policy,rollout_horizon):
    start_state = np.array([-1,1])
    tree = mcts.MonteCarloTree(builder.transition_function,
                               builder.cost_function,
                               builder.discount,
                               builder.actions,
                               policy,
                               value_fn,
                               start_state,
                               rollout_horizon)
    return tree

def run_tree(tree,N):
    for i in xrange(N):
        print '-'*10
        print 'Iter',i
        (path,a_list) = tree.path_to_leaf()
        leaf = path[-1]
        
        (G,a_id,state,cost) = tree.rollout(leaf.state)
        a_list.append(a_id)
        
        #assert(len(a_list) == len(path))
        if len(path) > 4:
            print 'Action prefix:',a_list[:3]
        tree.backup(path,a_list,G)
        print 'Value:',tree.root_node.V

        #mcts.display_tree(tree.root_node,
        #                  title='Iteration '+str(i))

(builder,mdp_obj) = make_di_mdp()
x = solve_mdp_kojima(mdp_obj)
(v_fn,q_fns) = make_interps(builder.discretizer,x)
q_policy = mdp.policy.MaxFunPolicy(mdp_obj.actions, q_fns)
tree = make_tree(builder,v_fn,q_policy,100)
run_tree(tree,10)
