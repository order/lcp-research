from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *

# This file is for pushing V and flow vectors to C++ code for simulation

root = 'data/di/' # root filename
disc_n = 20
action_n = 3

# Generate problem
step_len = 0.01
n_steps = 5
damp = 0.01
jitter = 0.1
discount = 0.99
bounds = [[-6,6],[-5,5]]
cost_radius = 0.25
actions = np.array([[-1],[0],[1]])
problem = make_di_problem(step_len,
                          n_steps,
                          damp,
                          jitter,
                          discount,
                          bounds,
                          cost_radius,
                          actions)

# Generate MDP
(mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)

# Solve
(p,d) = solve_with_kojima(mdp,1e-8,1000)

# Build value function
S = block_solution(mdp,p)

# Marshal up and ship off

marsh = Marshaller()
# Physics params
marsh.add(step_len)
marsh.add(n_steps)
marsh.add(damp)
marsh.add(jitter)

# Grid
marsh.add(np.array(bounds[0],dtype=np.double)) # low
marsh.add(np.array(bounds[1],dtype=np.double)) # high
marsh.add(disc_n*np.ones(2,dtype=np.double)) # num cells per dim

marsh.add(cost_radius)
marsh.add(discount)
print actions
marsh.add(actions)

marsh.save('test.mcts')


