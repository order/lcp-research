from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *

import cdiscrete as cd

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
(v,F) = split_solution(mdp,p)
Q = q_vectors(mdp,v);

(low,high,num) = zip(*disc.grid_desc)
low = np.array(low,dtype=np.double);
high = np.array(high,dtype=np.double);
num = np.array(num,dtype=np.uint64);
actions = mdp.actions.astype(np.double)

print 'Calling C++ function'
cd.mcts_test(Q,F,actions,low,high,num)
