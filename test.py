from utils.pickle import dump, load
from generate_di_problem import make_di_problem
from generate_mdp import make_uniform_mdp
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
root = 'data/di'
disc_n = 20
action_n = 3
type_policy = 'hand'
batch_size = 50
num_start_states = 31*batch_size
batch = True
horizon = 1


# Generate problem
problem = make_di_problem()

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)

# Build value function
(v,flow) = split_solution(mdp,p)
v_fn = InterpolatedFunction(disc,v)

# Build policies
policies = {}
q = q_vectors(mdp,v)
q_fns = build_functions(mdp,disc,q)
#policies['q'] = IndexPolicyWrapper(MinFunPolicy(q_fns),
#                                   mdp.actions)
flow_fns = build_functions(mdp,disc,flow)
#policies['flow'] = IndexPolicyWrapper(MaxFunPolicy(flow_fns),
#                                   mdp.actions)

initial_prob = probs.FunctionProbability(flow_fns)
#initial_prob = probs.UniformProbability(3)

bang_index_policy = BangBangPolicy()
bang_policy = IndexPolicyWrapper(bang_index_policy,
                                 mdp.actions)
#policies['bangbang'] = bang_policy
for epsilon in [0]:
    rollout_policy = EpsilonFuzzedPolicy(3,epsilon,
                                         bang_index_policy)
    name = 'bang_{0}'.format(epsilon)
    for rollout in [15]:
        for budget in [50]:
            prob_scale = 10
            name = 'mcts_{0}_{1}_{2}'.format(epsilon,
                                             rollout,
                                             budget)
            policies[name] = MCTSPolicy(problem,
                                        mdp.actions,
                                        rollout_policy,
                                        initial_prob,
                                        v_fn,
                                        rollout,
                                        prob_scale,
                                        budget) 

# Build start states
#start_states = problem.gen_model.boundary.random_points(
#    num_start_states)
#start_states = linalg.random_points([(-2,2),(-2,2)],
#                                    num_start_states)
N = num_start_states
n = int(N / 2)
start_states = np.empty((N,2))


x = 4*np.random.rand(n)-2
v = -np.sign(x) * np.sqrt(2 * np.abs(x)) + 0.2*np.random.rand(n) - 0.1
v = np.maximum(-2,np.minimum(2,v))
start_states[:n,0] = x
start_states[:n,1] = v
start_states[n:,:] = 4*np.random.rand(N-n,2) - 2
assert((num_start_states,2) == start_states.shape)


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
