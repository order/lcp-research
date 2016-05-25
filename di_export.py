from mdp import *

root = 'data/di'
disc_n = 20
action_n = 3
batch = True
horizon = 1


# Generate problem
problem = make_di_problem()

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)

# Build value function
S = block_solution(mdp,p)
save_ndarray_hdf5(root + '.solver_data.h5',S)
