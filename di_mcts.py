import numpy as np
import scipy as sp
import scipy.sparse as sps
import multiprocessing

import matplotlib.pyplot as plt

from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *
from lcp import *

from utils import *

from experiment import *

WORKERS = multiprocessing.cpu_count()-1
BATCHES_PER_WORKER = 5
STATES_PER_BATCH = 20
SIM_HORIZON = 500
MCTS_BUDGET = 1
ACTION_SELECT_MODE = ACTION_ROLLOUT

ROOT = os.path.expanduser('~/data/di') # root filename
DRIVER = os.path.expanduser('~/repo/lcp-research/cdiscrete/driver')
SAVE_FILE = os.path.expanduser('~/data/di_rollout.py')
def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.995          # Discount (\gamma)
    B = 5
    bounds = [[-B,B],[-B,B]]  # Square bounds, 
    cost_radius = 0.25        # Goal region radius
    
    actions = np.array([[-1],[0],[1]]) # Actions
    action_n = 3
    assert(actions.shape[0] == action_n)
    
    problem = make_di_problem(step_len,
                              n_steps,
                              damp,
                              jitter,
                              discount,
                              bounds,
                              cost_radius,
                              actions) # Just needs the action boundaries
    # Generate MDP
    (mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)
    assert(np.all(mdp.actions == actions)) # Should be the same
    #add_drain(mdp,disc,np.zeros(2),0)
    return (mdp,disc,problem)

def solve_mdp_with_kojima(mdp):
    # Solve
    start = time.time()
    (p,d) = solve_with_kojima(mdp,1e-8,1000)
    print 'Kojima ran for:', time.time() - start, 's'
    return (p,d)

#######################################################
# Driver start

# Build the MDP and discretizer
(mdp,disc,problem) = build_problem(16)
(ref_mdp,ref_disc,ref_problem) = build_problem(64)

# Solve, initially, using Kojima

if True:
    (ref_p,_) = solve_mdp_with_kojima(ref_mdp)
    (ref_v,_) = split_solution(ref_mdp,ref_p)
    
    (p,_) = solve_mdp_with_kojima(mdp)
    sol = block_solution(mdp,p)
    
    np.save('sol.npy', sol)
    np.save('ref_v.npy', ref_v)
else:
    sol = np.load('sol.npy')
    ref_v = np.load('ref_v.npy')   

mcts_params = MCTSParams(MCTS_BUDGET)
mcts_params.action_select_mode = ACTION_SELECT_MODE
fileroot = ROOT + '/di.mcts'
if True:
    start_states = create_start_states(STATES_PER_BATCH,
                                       problem,
                                       WORKERS * BATCHES_PER_WORKER)
    returns = get_mcts_returns(DRIVER,
                               fileroot,
                               problem,
                               mdp,
                               disc,
                               sol,
                               ref_disc,
                               ref_v,
                               mcts_params,
                               start_states,
                               SIM_HORIZON,
                               WORKERS)
    np.save(SAVE_FILE,returns)        

else:
    start_states = 4 * (2 * np.random.rand(25,2) - 1)
    build_di_mcts_file('cdiscrete/test.mcts',
                       problem,
                       mdp,
                       disc,
                       sol,
                       ref_disc,
                       ref_v,
                       mcts_params,
                       start_states,
                       SIM_HORIZON)

