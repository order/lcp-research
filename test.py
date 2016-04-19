from utils.pickle import dump, load
from generate_di_problem import make_di_problem
from generate_mdp import make_uniform_mdp
from solve_mdp_kojima import solve_with_kojima
from mdp.state_functions import InterpolatedFunction
from mdp.solution_process import *
from mdp.policy import MinFunPolicy,\
    IndexPolicyWrapper,BangBangPolicy
from mdp.simulator import *

import matplotlib.pyplot as plt
import utils.plotting


root = 'data/di'
disc_n = 20
action_n = 3
type_policy = 'hand'
num_start_states = 500
horizon = 2500

# Generate problem
problem = make_di_problem()
dump(problem,root+'.prob.pickle')

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)
dump(mdp,root+'.mdp.pickle')
dump(disc,root+'.disc.pickle')

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)
dump(p, root+'.psol.pickle')

# Build value function
(v,flow) = split_solution(mdp,p)
v_fn = InterpolatedFunction(disc,v)
dump(v_fn,root+'.vfun.pickle')

# Build policies
policies = {}
q = q_vectors(mdp,v)
q_fns = build_functions(mdp,disc,q)
policies['q'] = IndexPolicyWrapper(MinFunPolicy(q_fns),
                                   mdp.actions)

policies['handcrafted'] = BangBangPolicy()
dump(policies,root+'.policies.pickle')

# Build start states
start_states = problem.gen_model.boundary.random_points(
    num_start_states)
dump(start_states,root+'.start_states.pickle')

# Simulate
results = {}
for (name,policy) in policies.items():
    result = simulate(problem,
                      policy,
                      start_states,
                      horizon)
    results[name] = result
dump(results,root+'.sim.pickle')

# Return
returns = {}
for (name,result) in results.items():
    (action,states,costs) = result
    returns[name] = discounted_return(costs,
                                      problem.discount)
dump(returns,root+'.returns.pickle')

# V
vals = v_fn.evaluate(start_states)
dump(vals,root+'.vals.pickle')

quit()

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
