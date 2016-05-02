import numpy as np
import matplotlib.pyplot as plt

from utils import *

import config
from mdp import *
from mdp.policies import *
import mcts

import linalg
import time

writetodisk = True
root = 'data/hallway'
nodes = 50
action_n = 3
type_policy = 'hand'
batch_size = 10
num_start_states = 3*batch_size
batch = True
horizon = 70

rebuild_all = False
build_problem = False
build_mdp = False
run_solver = False

# Build problem
prob_build_fn = lambda: config.mdp.make_hallway_problem(nodes)
prob_file = root + '.prob.pickle'
problem = do_or_load(prob_build_fn,
                     prob_file,
                     rebuild_all or build_problem,
                     'problem build')


# Generate MDP and discretizer
actions = np.array([-1,0,1]).reshape((3,1))
mdp_build_fn = lambda: config.mdp.make_trivial_mdp(problem,
                                                   nodes,
                                                   actions)
mdp_file = root + '.mdp.pickle'
(mdp_obj,disc) = do_or_load(mdp_build_fn,
                        mdp_file,
                        rebuild_all or build_mdp,
                        'mdp build')

# Solve with Kojima
solve_fn = lambda: config.solver.solve_with_kojima(mdp_obj,
                                                   1e-8,1000)
soln_file = root + 'soln.pickle'
(p,d) = do_or_load(solve_fn,
                   soln_file,
                   rebuild_all or run_solver,
                   'solver')

# Build value function
print 'Building value function'
(v,flow) = split_solution(mdp_obj,p)
v_fn = InterpolatedFunction(disc,v)
dump(v_fn,root + '.vfn.pickle')

# Build flow functions
flow_fns = build_functions(mdp_obj,disc,flow)

#######################
# Build policies
policy_dict = {}
for _ in xrange(250): 
    scale = 0.5*np.random.rand() + 0.1
    q = q_vectors(mdp_obj,v + scale*np.random.randn(v.size))
    q_fns = build_functions(mdp_obj,disc,q)
    policy_dict[scale] = IndexPolicyWrapper(MinFunPolicy(q_fns),
                                         mdp_obj.actions)
"""
print 'Building policies'
q = q_vectors(mdp_obj,v)
q_fns = build_functions(mdp_obj,disc,q)
policy_dict['q'] = IndexPolicyWrapper(MinFunPolicy(q_fns),
                                       mdp_obj.actions)
policy_dict['hand'] = HallwayPolicy(nodes)

rollout_policy = HallwayPolicy(nodes)
initial_prob = probs.FunctionProbability(flow_fns)
budget = 10
rollout = 5
prob_scale = 10
policy_dict['mcts'] = mcts.MCTSPolicy(problem,
                                   mdp_obj.actions,
                                   rollout_policy,
                                   initial_prob,
                                   v_fn,
                                   rollout,
                                   prob_scale,
                                   budget) 
"""
start_states = np.random.randint(nodes,
                                 size=(num_start_states,1))
dump(start_states,root + '.starts.pickle')

# Simulate
print 'Simulating'
results = {}
start = time.time()
for (name,policy) in policy_dict.items():
    print '\tRunning {0} jobs'.format(name)
    if batch:
        result = batch_simulate(problem,
                                policy,
                                start_states,
                                horizon,
                                batch_size)
    else:
        result = simulate(problem,
                          policy,
                          start_states,
                          horizon)
    assert((num_start_states,horizon) == result.costs.shape)
    results[name] = result
print '**Multithread total', time.time() - start
dump(results,root + '.results.pickle')

# Return
returns = {}
for (name,result) in results.items():
    returns[name] = discounted_return(result.costs,
                                      problem.discount)
    assert((num_start_states,) == returns[name].shape)
dump(returns,root + '.return.pickle')

# V
vals = v_fn.evaluate(start_states)
dump(vals,root + '.vals.pickle')
