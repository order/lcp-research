from utils.pickle import dump, load
from generate_hallway_problem import make_hallway_problem
from generate_mdp import make_trivial_mdp
from solve_mdp_kojima import solve_with_kojima
from mdp.state_functions import InterpolatedFunction
from mdp.solution_process import *
from mdp.policy import MinFunPolicy,MaxFunPolicy,\
    IndexPolicyWrapper,BangBangPolicy,EpsilonFuzzedPolicy
import mdp.probs as probs
from mcts import MCTSPolicy
from mdp.simulator import *

import linalg
import matplotlib.pyplot as plt
import utils.plotting
import time

writetodisk = True
root = 'data/hallway'
nodes = 50
action_n = 3
type_policy = 'hand'
batch_size = 1
num_start_states = 1
batch = False
horizon = 1


# Generate problem
problem = make_hallway_problem(nodes)

# Generate MDP
actions = np.array([-1,0,1]).reshape((3,1))
(mdp,disc) = make_trivial_mdp(problem,nodes,actions)

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)

# Build value function
(v,flow) = split_solution(mdp,p)
v_fn = InterpolatedFunction(disc,v)

plt.plot(v / np.max(v))
for i in xrange(3):
    plt.plot(flow[:,i] / np.max(flow[:,i]))
plt.show()

# Build policies
policies = {}
q = q_vectors(mdp,v)
q_fns = build_functions(mdp,disc,q)
policies['q'] = IndexPolicyWrapper(MinFunPolicy(q_fns),
                                   mdp.actions)

start_states = np.random.randint(nodes,
                                 size=(num_start_states,1))

# Simulate
results = {}
start = time.time()
for (name,policy) in policies.items():
    print 'Running {0} jobs'.format(name)
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

# Return
returns = {}
for (name,result) in results.items():
    returns[name] = discounted_return(result.costs,
                                      problem.discount)
# V
vals = v_fn.evaluate(start_states)

if writetodisk:
    """
    Dump stuff...
    """
    dump(problem,root+'.prob.pickle')
    dump(mdp,root+'.mdp.pickle')
    dump(disc,root+'.disc.pickle')
    dump(p, root+'.psol.pickle')
    dump(v_fn,root+'.vfun.pickle')
    dump(policies,root+'.policies.pickle')
    dump(start_states,root+'.start_states.pickle')
    dump(results,root+'.sim.pickle')
    dump(returns,root+'.returns.pickle')
    dump(vals,root+'.vals.pickle')
