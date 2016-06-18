import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt

from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *
from lcp import *

def build_problem():
    disc_n = 16 # Number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.95           # Discount (\gamma)
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
                              actions)
    # Generate MDP
    (mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)
    #add_drain(mdp,disc,np.zeros(2),0)
    return (mdp,disc)

def solve_mdp_with_kojima(mdp):
    # Solve
    start = time.time()
    (p,d) = solve_with_kojima(mdp,1e-12,1000)
    print 'Kojima ran for:', time.time() - start, 's'
    # Build value function
    return (p,d)

def build_projective_lcp(mdp,basis):
    start = time.time()
    # Build the LCP
    lcp = mdp.build_lcp()

    # Convert matrices to sparse and elim zeros
    B = sps.csr_matrix(basis)
    B.eliminate_zeros()
    M = sps.csr_matrix(lcp.M)
    M.eliminate_zeros()

    # Project MDP onto basis
    PtPU = B.T.dot(M)
    plcp = ProjectiveLCPObj(B, PtPU, lcp.q)    
    print 'Building projective LCP: {0}s'.format(time.time() - start)
    return plcp

def solve_mdp_with_projective(mdp,basis,p,d):
    plcp = build_projective_lcp(mdp,basis)
    start = time.time()

    (N,k) = basis.shape
    if False:
        x0 = np.maximum(1e-15,p)
        y0 = np.maximum(1e-15,d)
        w0 = basis.T.dot(x0 - y0 + plcp.q)
    else:
        x0 = np.ones(N)
        y0 = np.ones(N)
        w0 = np.ones(k)        
        
    (p,d) = solve_with_projective(plcp,1e-12,2500,x0,y0,w0)
    print 'Projective ran for:', time.time() - start, 's'
    return block_solution(mdp,p)

def get_basis_from_array(mdp,disc,f,num_bases):
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

def get_basis_from_solution(mdp,disc,sol,num_bases):
    (N,Ap) = sol.shape
    assert(N == mdp.num_states)
    assert(Ap == mdp.num_actions+1)

    # Find good bases for each of the vectors
    Bases = []
    total_bases = 0
    for i in xrange(Ap):
        A = reshape_full(sol[:,i],disc)
        if False or i != 0:
            B = get_basis_from_array(mdp,disc,A,num_bases)
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

#######################################################
# Driver start

# Build the MDP and discretizer
(mdp,disc) = build_problem()

# Solve, initially, using Kojima

start = time.time()
(p,d) = solve_mdp_with_kojima(mdp)
sol = block_solution(mdp,p)

ktime = time.time() - start

if True:
    for i in xrange(mdp.num_actions + 1):
        plt.subplot(3,4,i+1)
        F = reshape_full(sol[:,i],disc)
        imshow(F)

# Build basis from useful Fourier waves
basis = get_basis_from_solution(mdp,disc,sol,60)
low_p = basis.dot(basis.T.dot(p))
low_d = basis.dot(basis.T.dot(d))

low_sol = block_solution(mdp,low_p)

if True:
    for i in xrange(mdp.num_actions + 1):
        plt.subplot(3,4,i+5)
        F = reshape_full(low_sol[:,i],disc)
        imshow(F)

# Solve with projective method
start = time.time()
p_sol = solve_mdp_with_projective(mdp,basis,low_p,low_d)
ptime = time.time() - start

print 'Time ratio:', ptime / ktime

if True:
    for i in xrange(mdp.num_actions + 1):
        plt.subplot(3,4,i+9)
        F = reshape_full(p_sol[:,i],disc)
        imshow(F)
plt.show()
