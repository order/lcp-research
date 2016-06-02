from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *

import math

# This file is for pushing V and flow vectors to C++ code for simulation

root = 'data/di/' # root filename

#####################################
# Problem parameters

disc_n = 20 # Number of cells per dimension

step_len = 0.01           # Step length
n_steps = 5               # Steps per iteration
damp = 0.01               # Dampening
jitter = 0.1              # Control jitter 
discount = 0.99           # Discount (\gamma)
B = 5 
bounds = [[-B,B],[-B,B]]  # Square bounds, 
cost_radius = 0.25        # Goal region radius

actions = np.array([[-1],
                    [0],
                    [1]]) # Actions
action_n = 3
assert(actions.shape[0] == action_n)

p_scale = 1
ucb_scale = 1
rollout_horizon = 75
mcts_budget = 2500
tail_error = 1
sim_horizon = int(math.log(tail_error*(1.0 - discount)) / math.log(discount))

# Uniform start states
start_states = B*(2*np.random.rand(10000,2) - 1)

#########################################
# Generate stuff
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
(p,d) = solve_with_kojima(mdp,1e-12,1000)

# Build value function
(v,flow) = split_solution(mdp,p)
assert(np.all(flow > 0))
q = q_vectors(mdp,v)
assert(np.all(q > 0))

########################################
# Marshal up and ship off

marsh = Marshaller()

# Grid
marsh.add(-B*np.ones(2,dtype=np.double)) # low
marsh.add(B*np.ones(2,dtype=np.double)) # high
marsh.add(disc_n*np.ones(2,dtype=np.double)) # num cells per dim

# Physics params
marsh.add(step_len)
marsh.add(n_steps)
marsh.add(damp)
marsh.add(jitter)

# Other MDP params
marsh.add(cost_radius)
marsh.add(discount)
marsh.add(actions)

# MCTS context
marsh.add(v)
marsh.add(q)
marsh.add(flow)

marsh.add(p_scale)
marsh.add(ucb_scale)
marsh.add(rollout_horizon)

marsh.add(mcts_budget)

# Simulation
marsh.add(sim_horizon)
marsh.add(start_states)

print "Marshalling", len(marsh.objects),"objects"

marsh.save('cdiscrete/test.mcts')


