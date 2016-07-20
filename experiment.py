import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt

import multiprocessing
import subprocess
from mdp import *
from mdp.policies import *
from mdp.transitions import *
from utils import Marshaller

from discrete import make_points

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

PROBLEM_DI = 1
PROBLEM_HILLCAR = 2

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

##########################################
# Plot the data dictionary
def plot_data_dict(data):
    single_keys=[]
    for (key,value) in data.items():
        D = np.array(value)
        if (2 == len(D.shape)):
            plt.figure()
            plt.semilogy(D,'-b',alpha=0.5)
            plt.title(key)
        else:
            single_keys.append(key)
    plt.figure()
    for key in single_keys:
        plt.semilogy(np.array(data[key]))
    plt.legend(single_keys,loc='best')
    plt.show

##########################################
# Plot the data dictionary
def plot_data_dict_abs_diff(A,B):
    A_keys = set(A.keys())
    B_keys = set(B.keys())

    ignoring = list(A_keys ^ B_keys)
    if len(ignoring) > 0:
        print 'Ignoring',ignoring

    overlap = A_keys & B_keys
    single_keys=[]
    for key in overlap:
        D = np.abs(np.array(A[key]) - np.array(B[key])) + 1e-35
        if (2 == len(D.shape)):
            plt.figure()
            plt.semilogy(D,'-b',alpha=0.5)
            plt.title(key)
        else:
            single_keys.append(key)
    plt.figure()
    for key in single_keys:
        D = np.abs(np.array(A[key]) - np.array(B[key])) + 1e-35
        plt.semilogy(D)
    plt.legend(single_keys,loc='best')
    plt.show

############################################
# Plot the solution in (A+1) plots
def plot_sol_images_interp(mdp,disc,x,G=512):
    A = mdp.num_actions
    blocks = block_solution(mdp,x)

    R = int(np.ceil(np.sqrt(A+1)))
    C = int(np.ceil(float(A+1) / float(R)))

    low = disc.get_lower_boundary()
    hi = disc.get_upper_boundary()
    assert(2 == len(low))

    [P,[X,Y]] = make_points([np.linspace(low[0],hi[0],G),
                             np.linspace(low[1],hi[1],G)],True)
    plt.figure()
    # Value block    
    plt.subplot(R,C,1)
    v_fn = InterpolatedFunction(disc,blocks[:,0])
    Z = np.reshape(v_fn.evaluate(P),X.shape)
    plt.pcolormesh(X,Y,Z,cmap = 'jet')
    plt.title('Value')
    plt.colorbar()

    for i in xrange(1,A+1):
        plt.subplot(R,C,i+1)
        f_fn = InterpolatedFunction(disc,blocks[:,i])
        Z = np.reshape(f_fn.evaluate(P),X.shape)
        Z = np.log(Z + 1e-20)
        plt.pcolormesh(X,Y,Z,cmap = 'plasma')
        plt.title('log Action ' + str(i-1))
        plt.colorbar()

def plot_sol_images(mdp,disc,x):
    A = mdp.num_actions
    blocks = block_solution(mdp,x)

    R = int(np.ceil(np.sqrt(A+1)))
    C = int(np.ceil(float(A+1) / float(R)))
    
    plt.figure()
    # Value block
    
    plt.subplot(R,C,1)
    img = reshape_full(blocks[:,0],disc)
    plt.pcolormesh(img,cmap = 'jet')
    plt.title('Value')
    plt.colorbar()

    for i in xrange(1,A+1):
        plt.subplot(R,C,i+1)
        img = reshape_full(blocks[:,i],disc)
        plt.pcolormesh(img,cmap = 'plasma')
        plt.title('Action ' + str(i-1))
        plt.colorbar()

def report_spectral_info(M):
    if isinstance(M,sps.spmatrix):
        M = M.toarray()
        
    [U,S,Vt] = np.linalg.svd(M)
    print "\tSV(M):", (S[-1],S[0])
    
    [U,S,Vt] = np.linalg.svd((M+M.T))
    print "\tSV(M + Mt):", (S[-1],S[0])
    
    [L,X] = np.linalg.eigh(M.T.dot(M))
    print "\tEV(MtM):", (L[0],L[-1])
    
    print '\tMonotone:', S[-1]/2.0
    print '\tLipschitz:', np.sqrt(L[-1])
    print '\tAlpha:', S[-1]/(2.0 * L[-1])
        
    
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

def contour_features(f,k):
    (N,) = f.shape

    # Form percentiles, discard 0 and 100
    P = np.percentile(f,np.linspace(0,100,k+1))
    assert((k+1,) == P.shape)

    B = np.empty((N,k))
    for i in xrange(k-1):
        B[:,i] = (P[i] <= f) * (f < P[i+1])
    B[:,k-1] = (P[k-1] <= f) * (f <= P[k])

    return B

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
##################################################
# Build a trig basis that explains f well
def get_trig_basis_from_block(mdp_obj,disc,f,num_bases):
    (n,) = f.shape
    assert(n == disc.num_nodes())
    
    F = reshape_full(f,disc)

    # Use the real FFT to find some reasonable bases
    (freq,shift,_) = top_trig_features(F,num_bases,1e-8)
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

def get_contour_basis_from_block(disc,f,num_bases):
    (n,) = f.shape
    K = num_bases
    assert(n == disc.num_nodes())

    P = np.percentile(f,np.linspace(0,100,K+1))
    assert(P[0] == np.min(f))
    assert(P[-1] == np.max(f))

    B = np.zeros((n,K),dtype=np.double)

    # Add ones
    for i in xrange(K):
        idx = (P[i] <= f)
        B[idx,i] = 1

    assert(np.all(np.all(B >= 0)))
    
    return B

def get_jigsaw_basis_from_block(disc,f,num_bases):
    (n,) = f.shape
    K = num_bases
    assert(n == disc.num_nodes())

    P = np.percentile(f,np.linspace(0,100,K+1))

    B = np.zeros((n,K+1),dtype=np.double)

    # Add ones
    for i in xrange(K):
        idx = np.logical_and(P[i] <= f, f < P[i+1])
        B[idx,i] = f[idx] + 1e-3
    idx = (f == P[K])
    B[idx,K] = f[idx]

    assert(np.all(np.all(B >= 0)))
    
    return B

def get_solution_and_noise_basis(mdp,p,d,num_basis):
    (n,k) = p.shape
    columns = [np.ones((n,1)),p,d]
    columns.extend([x[:,np.newaxis] for x in mdp.costs])
    columns = np.hstack(columns)
    assert(num_basis >= columns.shape[1])
    return np.hstack([columns,
                      np.random.rand(n,num_basis-columns.shape[1])])   

###############################################################
# Use the above routine to build a basis for the entire problem
def get_basis_from_solution(mdp_obj,
                            disc,
                            primal_sol,
                            dual_sol,
                            mode,
                            num_bases):

    # Check mode
    mode = mode.lower()
    assert(mode in ['identity',
                    'trig',
                    'contour',
                    'jigsaw',
                    'solution'])
        
    (N,Ap) = primal_sol.shape
    assert(N == mdp_obj.num_states)
    assert(Ap == mdp_obj.num_actions+1)

    # Find good bases for each of the vectors
    Bases = []
    total_bases = 0
    for i in xrange(Ap):
        if 'trig' == mode:
            B = get_trig_basis_from_block(mdp_obj,
                                          disc,
                                          primal_sol[:,i],
                                          num_bases)
        elif 'contour' == mode:
            B = get_contour_basis_from_block(disc,
                                             primal_sol[:,i],
                                             num_bases)
        elif 'jigsaw' == mode:
            B = get_jigsaw_basis_from_block(disc,
                                            primal_sol[:,i],
                                            num_bases)
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

def build_mcts_file(filename,
                    problem,
                    mdp_obj,
                    disc,
                    mcts_sol,
                    ref_disc,
                    ref_v,
                    mcts_params,
                    start_states,
                    simulation_horizon):
    t_fn = problem.gen_model.trans_fn
    if isinstance(t_fn,DoubleIntegratorTransitionFunction):
        build_di_mcts_file(filename,
                           problem,
                           mdp_obj,
                           disc,
                           mcts_sol,
                           ref_disc,
                           ref_v,
                           mcts_params,
                           start_states,
                           simulation_horizon)
    elif isinstance(t_fn,HillcarTransitionFunction):
        build_hillcar_mcts_file(filename,
                                problem,
                                mdp_obj,
                                disc,
                                mcts_sol,
                                ref_disc,
                                ref_v,
                                mcts_params,
                                start_states,
                                simulation_horizon)
    else:
        raise NotImplementedError()
        

###############################################################
# Build a file for C++ MCTS with double integrator.
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
    marsh.add(PROBLEM_DI)
    
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

###############################################################
# Build a file for C++ MCTS solver with HILLCAR transfer function
# NB: changes here need to be synchronized with C++ code.
def build_hillcar_mcts_file(filename,
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
    marsh.add(PROBLEM_HILLCAR)
    
    # Boundary and discretization
    marsh.add(disc.get_lower_boundary())
    marsh.add(disc.get_upper_boundary())
    marsh.add(disc.get_num_cells())

    # Transition function parameters
    gen_model = problem.gen_model
    trans_fn = gen_model.trans_fn
    assert(isinstance(trans_fn,HillcarTransitionFunction))
    marsh.add(trans_fn.step)
    marsh.add(trans_fn.num_steps)
    marsh.add(trans_fn.dampening)
    marsh.add(trans_fn.jitter)

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

################################################
# Basic simulation
def get_returns(problem,
                policy,
                ref_v_fn,
                start_states,
                sim_horizon):

    (N,d) = start_states.shape
    sim_res = simulate(problem,
                       policy,
                       start_states,
                       sim_horizon)

    # Get total return using reference v function
    ret = discounted_return_with_tail_estimate(problem,
                                               sim_res.costs,
                                               sim_res.states,
                                               problem.discount,
                                               ref_v_fn)
    return (ret,sim_res)

#################################################
# Simulate Q policy

def build_q_policy(q_mdp,q_disc,v):
    q = q_vectors(q_mdp,v)
    q_fns = InterpolatedMultiFunction(q_disc,q)
    q_idx_policy = MinFunPolicy(q_fns)
    q_policy = IndexPolicyWrapper(q_idx_policy,
                                  q_mdp.actions)
    return q_policy

def get_q_returns(problem,
                  q_mdp,
                  q_disc,
                  v,
                  ref_v_fn,
                  start_states,
                  sim_horizon):
    # Build the Q policy
    q_policy = build_q_policy(q_mdp,q_disc,v)
    return get_returns(problem,
                       q_policy,
                       ref_v_fn,
                       start_states,
                       sim_horizon)



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

    # Should be a list of np.ndarray
    assert(isinstance(start_states,list))
    
    # Write out config files
    files = []
    for start in start_states:
        filename = root_filename + '.' + str(get_mcts_returns.FILE_NUMBER)
        get_mcts_returns.FILE_NUMBER += 1
        build_mcts_file(filename,
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
