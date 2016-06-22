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

#WORKERS = 1
WORKERS = multiprocessing.cpu_count()-1
BATCHES_PER_WORKER = 5
STATES_PER_BATCH = 5
SIM_HORIZON = 1000
BUILD_MODE = 'load'
BUILD_MODE = 'build'

low_dim = 16
ref_dim = 64


BUDGETS = [4,8,16,32,64,128,256,512,1024,2048]
#BUDGETS = [8]


ROOT = os.path.expanduser('~/data/hillcar') # root filename
DRIVER = os.path.expanduser('~/repo/lcp-research/cdiscrete/driver')
SAVE_FILE = ROOT + '/hillcar.'
def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 2               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.05             # Control jitter 
    discount = 0.997          # Discount (\gamma)
    bounds = [[-2,6],[-4,4]]  # Square bounds, 
    cost_radius = 0.25        # Goal region radius
    
    actions = np.array([[-5],[0],[5]]) # Actions
    action_n = 3
    assert(actions.shape[0] == action_n)
    
    problem = make_hillcar_problem(step_len,
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
    (p,d) = solve_with_kojima(mdp,1e-8,1000,1e-8,1e-6)
    print 'Kojima ran for:', time.time() - start, 's'
    return (p,d)

if __name__ == '__main__':

    ####################################################
    # Build the MDP and discretizer
    (low_mdp,low_disc,problem) = build_problem(low_dim)
    (ref_mdp,ref_disc,_) = build_problem(ref_dim)
    
    ####################################################
    # Solve, initially, using Kojima
    if BUILD_MODE == 'build':
        # Build / load
        (ref_p,_) = solve_mdp_with_kojima(ref_mdp)
        ref_sol = block_solution(ref_mdp,ref_p)
        
        (low_p,_) = solve_mdp_with_kojima(low_mdp)
        low_sol = block_solution(low_mdp,low_p)
        
        np.save('low_sol.npy', low_sol)
        np.save('ref_sol.npy', ref_sol)
    else:
        assert(BUILD_MODE == 'load')
        low_sol = np.load('low_sol.npy')
        ref_sol = np.load('ref_sol.npy')

    ref_v = ref_sol[:,0]
    ref_v_fn = InterpolatedFunction(ref_disc,ref_v)

    ref_p = ref_sol.reshape(-1,order='F')
    low_v = low_sol[:,0]

    img = reshape_full(ref_v,ref_disc)
    imshow(img)
    plt.show()
    
    ####################################################
    # Form the Fourier projection (both value and flow)
    B = get_basis_from_solution(ref_mdp,ref_disc,ref_sol,low_dim**2)
    (N,K) = B.shape
    assert((N,) == ref_p.shape)

    proj_p = B.dot(B.T.dot(ref_p))
    proj_sol = block_solution(ref_mdp,proj_p)
    proj_v = proj_sol[:,0]


    #####################################################
    # Build the start states
    batched_start_states = create_start_states(STATES_PER_BATCH,
                                               problem,
                                               WORKERS * BATCHES_PER_WORKER)
    start_states = np.vstack(batched_start_states)

    start_states = np.array([[4.0,0.0]])
    batched_start_states = [start_states]

    #####################################################
    # Rollout
    rollout_policy = LinearPolicy(ref_mdp.actions,np.array([1,1]))
    (rollout_ret,sim) = get_returns(problem,
                                    rollout_policy,
                                    ref_v_fn,
                                    start_states,
                                    SIM_HORIZON)
    print 'Rollout policy:', np.percentile(rollout_ret,[25,50,75])
    np.save(SAVE_FILE + 'rollout',rollout_ret)

    """
    (N,d,T) = sim.states.shape
    for i in xrange(N):
        plt.plot(sim.states[i,0,:],
                 sim.states[i,1,:],
                 'x-k',alpha=0.5)
    plt.show()
    """
    
    #####################################################
    # Run Q-policy with coarse grid values
    (q_ret,sim) = get_q_returns(problem,
                                low_mdp,
                                low_disc,
                                low_v,
                                ref_v_fn,
                                start_states,
                                SIM_HORIZON)
    print 'Coarse Q-policy:',np.percentile(q_ret,[25,50,75])
    np.save(SAVE_FILE + 'q_low',q_ret)

    """
    (N,d,T) = sim.states.shape
    for i in xrange(N):
        plt.plot(sim.states[i,0,:],
                 sim.states[i,1,:],
                 'x-k',alpha=0.5)
    plt.show()
    """

    #####################################################
    # Run Q-policy with fine grid values
    q_ret,_ = get_q_returns(problem,
                          ref_mdp,
                          ref_disc,
                          ref_v,
                          ref_v_fn,
                          start_states,
                          SIM_HORIZON)
    print 'Fine Q-policy:',np.percentile(q_ret,[25,50,75])
    np.save(SAVE_FILE + 'q_ref',q_ret)

    (N,d,T) = sim.states.shape
    for i in xrange(N):
        plt.plot(sim.states[i,0,:],
                 sim.states[i,1,:],
                 'x-k',alpha=0.5)
    plt.show()
    quit()
    
    #####################################################
    # Run Q-policy with fourier values   
    q_ret,_ = get_q_returns(problem,
                          ref_mdp,
                          ref_disc,
                          proj_v,
                          ref_v_fn,
                          start_states,
                          SIM_HORIZON)
    print 'Fourier Q-policy:',np.percentile(q_ret,[25,50,75])

    np.save(SAVE_FILE + 'q_proj',q_ret)
    
        
    ####################################################
    # MCTS with Coarse Q
    for budget in BUDGETS:
        mcts_params = MCTSParams(budget)
        mcts_params.action_select_mode = ACTION_Q
        fileroot = ROOT + '/di.mcts'
        mcts_ret = get_mcts_returns(DRIVER,
                                    fileroot,
                                    problem,
                                    low_mdp,
                                    low_disc,
                                    low_sol,
                                    ref_disc,
                                    ref_v,
                                    mcts_params,
                                    batched_start_states,
                                    SIM_HORIZON,
                                    WORKERS)
        print 'MCTS Coarse policy ({0}):'.format(budget),\
            np.percentile(mcts_ret,[25,50,75])
        np.save(SAVE_FILE + 'mcts_low_{0}'.format(budget), mcts_ret)
    ####################################################
    # MCTS with Projected Q
    for budget in BUDGETS:
        mcts_params = MCTSParams(budget)
        mcts_params.action_select_mode = ACTION_Q
        fileroot = ROOT + '/di.mcts'
        mcts_ret = get_mcts_returns(DRIVER,
                                    fileroot,
                                    problem,
                                    ref_mdp,
                                    ref_disc,
                                    proj_sol,
                                    ref_disc,
                                    ref_v,
                                    mcts_params,
                                    batched_start_states,
                                    SIM_HORIZON,
                                    WORKERS)
        print 'MCTS projected policy ({0}):'.format(budget),\
            np.percentile(mcts_ret,[25,50,75])
        np.save(SAVE_FILE + 'mcts_proj_{0}'.format(budget), mcts_ret)

    ####################################################
    # MCTS with No Q
    noq_sol = np.array(low_sol)
    noq_sol[:,0] = 1.0/ (1.0 - problem.discount)
    for budget in BUDGETS:
        mcts_params = MCTSParams(budget)
        mcts_params.action_select_mode = ACTION_Q
        fileroot = ROOT + '/di.mcts'
        mcts_ret = get_mcts_returns(DRIVER,
                                    fileroot,
                                    problem,
                                    low_mdp,
                                    low_disc,
                                    noq_sol,
                                    ref_disc,
                                    ref_v,
                                    mcts_params,
                                    batched_start_states,
                                    SIM_HORIZON,
                                    WORKERS)
        print 'MCTS Coarse no Q policy ({0}):'.format(budget),\
            np.percentile(mcts_ret,[25,50,75])
        np.save(SAVE_FILE + 'mcts_noq_{0}'.format(budget), mcts_ret)
        
    ####################################################
    # MCTS with No Flow
    noflow_sol = np.array(low_sol)
    noflow_sol[:,1:] = 1.0
    for budget in BUDGETS:
        mcts_params = MCTSParams(budget)
        mcts_params.action_select_mode = ACTION_Q
        fileroot = ROOT + '/di.mcts'
        mcts_ret = get_mcts_returns(DRIVER,
                                    fileroot,
                                    problem,
                                    low_mdp,
                                    low_disc,
                                    noflow_sol,
                                    ref_disc,
                                    ref_v,
                                    mcts_params,
                                    batched_start_states,
                                    SIM_HORIZON,
                                    WORKERS)
        print 'MCTS Coarse no flow policy ({0}):'.format(budget),\
            np.percentile(mcts_ret,[25,50,75])
        np.save(SAVE_FILE + 'mcts_noflow_{0}'.format(budget), mcts_ret)

    ####################################################
    # MCTS with no MDP information
    neither_sol = np.empty(low_sol.shape)
    neither_sol[:,0] = 1.0/ (1.0 - problem.discount)
    neither_sol[:,1:] = 1.0
    for budget in BUDGETS:
        mcts_params = MCTSParams(budget)
        mcts_params.action_select_mode = ACTION_Q
        fileroot = ROOT + '/di.mcts'
        mcts_ret = get_mcts_returns(DRIVER,
                                    fileroot,
                                    problem,
                                    low_mdp,
                                    low_disc,
                                    neither_sol,
                                    ref_disc,
                                    ref_v,
                                    mcts_params,
                                    batched_start_states,
                                    SIM_HORIZON,
                                    WORKERS)
        print 'MCTS Coarse no mpd information policy ({0}):'.format(budget),\
            np.percentile(mcts_ret,[25,50,75])
        np.save(SAVE_FILE + 'mcts_noflow_{0}'.format(budget), mcts_ret)
