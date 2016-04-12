import numpy as np
import matplotlib.pyplot as plt

import time

from discrete.regular_interpolate import RegularGridInterpolator
import discrete.discretize import

from mdp.transition_functions.double_integrator\
    import DoubleIntegratorTransitionFunction

import mdp.policy
from mdp.state_functions import BallSetFn
from mdp.costs import CostWrapper
from mdp.mdp_builder import MDPBuilder
from mdp.boundary import SaturationBoundary
from mdp.generative_model import GenerativeModel


import solvers
from solvers.kojima import KojimaIPIterator
from solvers.value_iter import ValueIterator


import utils

import mcts

def make_di_mdp(N):

    trans_params = utils.kwargify(step=0.01,
                                  num_steps=5,
                                  dampening=0.01,
                                  control_jitter=0.01)
    trans_fn = DoubleIntegratorTransitionFunction(
        **trans_params)
    boundary = SaturationBoundary([(-5,5),(-5,5)])
    cost_state_fn = BallSetFn(np.zeros(2), 0.5)
    cost_fn = CostWrapper(cost_state_fn)
    state_dim = 2
    action_dim = 1 
    gen_model = GenerativeModel(trans_fn,
                                boundary,
                                cost_fn,
                                state_dim,
                                action_dim)

    discretizer = RegularGridInterpolator([(-5,5,N),
                                           (-5,5,N)])

    actions = np.array([[-1,0,1]]).T
    discount = 0.997
    num_samples = 10
    builder = mdp_builder.MDPBuilder(gen_model,
                                     discretizer,
                                     actions,
                                     discount,
                                     num_samples)
    mdp_obj = builder.build_mdp()

    return builder,mdp_obj

def solve_mdp_kojima(mdp_obj,reg,tol):
    lcp_obj = mdp_obj.build_lcp()
    iterator = KojimaIPIterator(lcp_obj,
                                val_reg=reg,
                                flow_reg=reg)
    
    it_solver = solvers.IterativeSolver(iterator)
    it_solver.termination_conditions.append(
        solvers.PrimalChangeTerminationCondition(tol))
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
    Vs = np.empty(N)
    for i in xrange(N):
        print '.'
        (path,a_list) = tree.path_to_leaf()
        leaf = path[-1]
        
        (G,a_id,state,cost) = tree.rollout(leaf.state)
        a_list.append(a_id)
        
        #assert(len(a_list) == len(path))
        tree.backup(path,a_list,G)
        Vs[i] = tree.root_node.V
    return Vs

    #mcts.display_tree(tree.root_node,
    #                  title='Iteration '+str(i))

# Good DI
(builder,mdp_obj) = make_di_mdp(80)
x = solve_mdp_kojima(mdp_obj,1e-12,1e-12)
(good_v_fn,good_q_fns) = make_interps(builder.discretizer,x)

# Cheap DI
(builder,mdp_obj) = make_di_mdp(20)
x = solve_mdp_kojima(mdp_obj,1e-8,1e-6)
(cheap_v_fn,cheap_q_fns) = make_interps(builder.discretizer,x)

#q_policy = mdp.policy.MaxFunPolicy(q_fns)
#tree_policy = mdp.policy.EpsilonFuzzedPolicy(len(q_fns),
#                                             0.05,
#                                             q_policy)
tree_policy = mdp.policy.SoftMaxFunPolicy(good_q_fns)
tree = make_tree(builder,good_v_fn,tree_policy,25)
H = 150
Vs = run_tree(tree,H)
v_good = good_v_fn.evaluate(tree.root_node.state[np.newaxis,:])[0]
v_cheap = cheap_v_fn.evaluate(tree.root_node.state[np.newaxis,:])[0]

plt.plot([0,H-1],[v_good,v_good],'--g',
         [0,H-1],[v_cheap,v_cheap],'--r',
         np.arange(H),Vs,'-b',lw=2.0)
plt.show()
