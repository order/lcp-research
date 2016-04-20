from utils.pickle import dump, load
from generate_di_problem import make_di_problem
from generate_mdp import make_uniform_mdp
from solve_mdp_kojima import solve_with_kojima
from mdp.state_functions import InterpolatedFunction
from mdp.solution_process import *
from mdp.policy import MinFunPolicy,MaxFunPolicy,\
    IndexPolicyWrapper,BangBangPolicy
from mcts import MCTSPolicy
from mdp.simulator import *

import matplotlib.pyplot as plt
import utils.plotting
import time

writetodisk = False
root = 'data/di'
disc_n = 20
action_n = 3
type_policy = 'hand'
num_start_states = 3000
horizon = 1000

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
#q = q_vectors(mdp,v)
#q_fns = build_functions(mdp,disc,q)
#policies['q'] = IndexPolicyWrapper(MinFunPolicy(q_fns),
#                                   mdp.actions)
#flow_fns = build_functions(mdp,disc,flow)
#policies['flow'] = IndexPolicyWrapper(MaxFunPolicy(flow_fns),
#                                      mdp.actions)
policies['handcrafted'] = BangBangPolicy()
"""
policies['mcts'] = MCTSPolicy(problem,
                              mdp.actions,
                              BangBangPolicy(),
                              v_fn,
                              5,
                              5)
"""

# Build start states
start_states = problem.gen_model.boundary.random_points(
    num_start_states)

# Simulate
results = {}
start = time.time()
for (name,policy) in policies.items():
    print 'Running {0} jobs'.format(name)
    result = batch_simulate(problem,
                            policy,
                            start_states,
                            horizon,
                            100,1)
    results[name] = result
print '**Single thread total', time.time() - start

start = time.time()
for (name,policy) in policies.items():
    print 'Running {0} jobs'.format(name)
    result = batch_simulate(problem,
                            policy,
                            start_states,
                            horizon,
                            100)
    results[name] = result
print '**Multithread total', time.time() - start
quit()

# Return
returns = {}
for (name,result) in results.items():
    (action,states,costs) = result
    returns[name] = discounted_return(costs,
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
   

plt.figure(1)
l = np.min(vals)
u = np.max(vals)
for ret in returns.values():
    plt.plot(vals,ret,'.')
plt.plot([l,u],[l,u],':r')
plt.xlabel('Expected')
plt.ylabel('Empirical')
plt.legend(returns.keys())

plt.figure(2)
for ret in returns.values():
    (xs,fs) = utils.plotting.cdf_points(ret)
    plt.plot(xs,fs)
plt.legend(returns.keys(),loc='best')
plt.show()