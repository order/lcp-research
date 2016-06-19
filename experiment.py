import numpy as np
import scipy.sparse as sps

import multiprocessing
import subprocess
from mdp import *
from mdp.transitions import DoubleIntegratorTransitionFunction
from utils import Marshaller

import os

#########################################
# Modes
UPDATE_RET_V = 1
UPDATE_RET_Q = 2
UPDATE_RET_GAIN = 4

ACTION_Q = 1
ACTION_FREQ = 2
ACTION_ROLLOUT = 3
ACTION_UCB = 4

########################################
# MCTS Parameter object
class MCTSParams(object):
    def __init__(self,budget):
        self.budget = budget
        self.p_scale = 5
        self.ucb_scale = 5
        self.rollout_horizon = 25
        self.q_min_step = 0.05
        self.update_return_mode = UPDATE_RET_Q
        self.action_select_mode = ACTION_Q

######################################################
# Use n-dim RFFT to extract good features
def top_k_value(q,k,thresh):
    assert(q.size > k > 0)
    sq = np.sort(q.flatten())
    N = len(sq)
    assert(N == q.size)
    
    for i in xrange(N-k-1,N):
        if sq[i] >= thresh:
            return sq[i]
    return sq[-1]

def top_trig_features(f,k,thresh):
    Ns = np.array(f.shape) # Get dimensions
    F = np.fft.rfftn(f) # Take real DFT

    # Get the threshold we need to filter at to get around k
    # basis functions
    Q = top_k_value(np.abs(F), min(k,F.size-1),thresh)

    # Iterate over entries. Better way of doing this?
    Niprod = 1.0 / np.product(Ns)
    coords = np.argwhere(np.abs(F) >= Q)
    (n,d) = coords.shape

    freq = []
    shift = []
    amp = []
    
    for i in xrange(n):
        coord = coords[i,:]
        tcoord = tuple(coord)
        R = np.real(F[tcoord])
        I = np.imag(F[tcoord])
        
        if np.abs(R) > thresh:
            freq.append(2*np.pi*coord)
            shift.append(0.5 * np.pi)
            if coord[0] == 0:
                a = R*Niprod
            else:
                a = 2*R*Niprod
            amp.append(a)

        if len(freq) >= k:
            break
            
        if np.abs(I) > 1e-12:
            freq.append(2*np.pi*coord)
            shift.append(0)
            if coord[0] == 0:
                a = -I*Niprod
            else:
                a = -2*I*Niprod
            amp.append(a)

        if len(freq) >= k:
            break

    freq  = np.array(freq)
    shift = np.array(shift)
    amp   = np.array(amp)

    return freq,shift,amp


def add_oob_nodes(B,k):
    # Adds columns and rows for oob nodes
    (N,K) = B.shape
    ExpandedB = np.zeros((N+k,K+k))
    ExpandedB[:N,:K] = B
    ExpandedB[N:,K:] = np.eye(k)

    #[[B 0]
    # [0 I]]
    return ExpandedB

def get_basis_from_array(mdp_obj,disc,f,num_bases):
    # Use the real FFT to find some reasonable bases
    (freq,shift,_) = top_trig_features(f,num_bases,1e-8)
    fn = TrigBasis(freq,shift)
    
    # Rescale so the functions are over the boundary,
    # rather than [0,1]*D
    fn.rescale(disc.grid_desc)

    # Evaluate and orthogonalize the basis
    # TODO: do this analytically... should be possible
    # but there is some odd aliasing that I don't understand.
    B = fn.get_orth_basis(disc.get_cutpoints())
    (N,k) = B.shape
    assert(N == disc.num_real_nodes())
    assert(N >= k)

    # Add additional non-phyiscal nodes for oob
    B = add_oob_nodes(B,disc.num_oob())
    (N,k) = B.shape
    assert(N == disc.num_nodes())
    assert(N >= k)
  
    return B

# Use the above routine to build a basis
def get_basis_from_solution(mdp_obj,disc,sol,num_bases):
    (N,Ap) = sol.shape
    assert(N == mdp_obj.num_states)
    assert(Ap == mdp_obj.num_actions+1)

    # Find good bases for each of the vectors
    Bases = []
    total_bases = 0
    for i in xrange(Ap):
        A = reshape_full(sol[:,i],disc)
        if False or i != 0:
            B = get_basis_from_array(mdp_obj,disc,A,num_bases)
        else:
            B = sps.eye(N)
        (n,k) = B.shape
        assert(n == N)
        #assert(k <= num_bases + disc.num_oob())
        total_bases += k
        Bases.append(B)

    # Stitch together
    BigB = sps.block_diag(Bases)
    return BigB

# Build a list of start state blocks; for use in multi-threaded solving
def create_start_states(N,problem,Batches):
    starts = []
    bound = problem.gen_model.boundary
    return [bound.random_points(N) for _ in xrange(Batches)]

###############################################################
# Build a file for C++ MCTS solver.
# NB: changes here need to be synchronized with C++ code.
def build_di_mcts_file(filename,
                       problem,
                       mdp_obj,
                       disc,
                       mcts_sol,
                       ref_disc,
                       ref_v,
                       mcts_params,
                       start_states,
                       simulation_horizon):
    
    marsh = Marshaller()

    # Boundary and discretization
    marsh.add(disc.get_lower_boundary())
    marsh.add(disc.get_upper_boundary())
    marsh.add(disc.get_num_cells())

    # Transition function parameters
    gen_model = problem.gen_model
    trans_fn = gen_model.trans_fn
    assert(isinstance(trans_fn,DoubleIntegratorTransitionFunction))
    marsh.add(trans_fn.step)
    marsh.add(trans_fn.num_steps)
    marsh.add(trans_fn.dampening)
    marsh.add(trans_fn.control_jitter)

    # Other problem parameters
    cost_fn = gen_model.cost_fn
    assert(isinstance(cost_fn,CostWrapper))
    cost_state_fn = cost_fn.state_fn
    assert(isinstance(cost_state_fn,BallSetFn))
    marsh.add(cost_state_fn.radius)
    marsh.add(problem.discount)
    marsh.add(mdp_obj.actions)

    # Value and flow to use within MCTS
    marsh.add(mcts_sol[:,0])  # Value
    marsh.add(mcts_sol[:,1:]) # Flow

    # MCTS parameters
    marsh.add(mcts_params.budget)
    marsh.add(mcts_params.p_scale)
    marsh.add(mcts_params.ucb_scale)
    marsh.add(mcts_params.rollout_horizon)
    marsh.add(mcts_params.q_min_step)
    marsh.add(mcts_params.update_return_mode)
    marsh.add(mcts_params.action_select_mode)

    # Simulation parameters
    marsh.add(simulation_horizon)
    marsh.add(start_states)

    marsh.add(ref_disc.get_lower_boundary())
    marsh.add(ref_disc.get_upper_boundary())
    marsh.add(ref_disc.get_num_cells())
    marsh.add(ref_v)

    marsh.save(filename)


##################################################
# From problem objects start a number of C++ MCTS
# solves using a multi-threaded pool.
def run_command(cmd):
    curproc = multiprocessing.current_process()
    devnull = open(os.devnull, 'w')
    try:
        ret = subprocess.check_output(
            cmd, shell=False)
            #stderr=devnull)
        return ret
    except Exception as e:
        print e
        quit()

def get_mcts_returns(driver,
                     root_filename,
                     problem,
                     mdp_obj,
                     disc,
                     mcts_sol,
                     ref_disc,
                     ref_v,
                     mcts_params,
                     start_states,
                     sim_horizon,
                     num_workers):



    if not hasattr(get_mcts_returns, 'FILE_NUMBER'):
        get_mcts_returns.FILE_NUMBER = 0
    
    # Write out config files
    files = []
    for start in start_states:
        filename = root_filename + '.' + str(get_mcts_returns.FILE_NUMBER)
        get_mcts_returns.FILE_NUMBER += 1
        build_di_mcts_file(filename,
                           problem,
                           mdp_obj,
                           disc,
                           mcts_sol,
                           ref_disc,
                           ref_v,
                           mcts_params,
                           start,
                           sim_horizon)
        files.append(filename)
    print 'Running {0} jobs on {1} workers'.format(len(start_states),
                                                   num_workers)

    # Simulate from config files
    pool = multiprocessing.Pool(num_workers)
    commands = zip([driver]*len(files),
                   files)
    ret = pool.map(run_command,commands)
    pool.close()
    pool.join()

    returns = np.array([map(float,x.split()) for x in ret]).flatten()
    return returns
