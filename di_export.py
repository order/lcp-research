from mdp import *
from config.mdp import *
from config.solver import *
from utils import *

# This file is for pushing V and flow vectors to C++ code for simulation

root = 'data/di/' # root filename
disc_n = 20
action_n = 3

# Generate problem
problem = make_di_problem()

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)

# Build value function
S = block_solution(mdp,p)
save_ndarray_hdf5(root + 'solver_data.h5',S)
