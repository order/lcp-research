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

DIM = 32

def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.05             # Control jitter 
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
    (p,d) = solve_with_kojima(mdp,1e-8,1000,1e-8,1e-6)
    print 'Kojima ran for:', time.time() - start, 's'
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
    lcp = mdp.build_lcp()
    plcp = build_projective_lcp(mdp,basis)
    start = time.time()

    (N,k) = basis.shape
    x0 = np.ones(N)
    y0 = np.maximum(lcp.M.dot(x0) + lcp.q,1e-2)
    
    w0 = np.maximum(basis.T.dot(x0 - y0 + plcp.q),1e-2)
    assert((k,) == w0.shape)
        
    (p,d) = solve_with_projective(plcp,1e-12,250,x0,y0,w0)
    print 'Projective ran for:', time.time() - start, 's'
    return block_solution(mdp,p)

if __name__ == '__main__':

    ####################################################
    # Build the MDP and discretizer
    (mdp,disc,_) = build_problem(DIM)
    
    ####################################################
    # Solve, initially, using Kojima
        # Build / load
    (p,d) = solve_mdp_with_kojima(mdp)
    sol = block_solution(mdp,p)

    ####################################################
    # Form the Fourier projection (both value and flow)
    B = get_basis_from_solution(mdp,disc,sol,150)
    #B = np.eye(p.size)
    print 'Basis shape:',B.shape

    
    # Solve with projective method
    start = time.time()
    p_sol = solve_mdp_with_projective(mdp,B,p,d)
    ptime = time.time() - start

    for i in xrange(4):
        plt.subplot(2,2,i+1)
        img = reshape_full(p_sol[:,i],disc)
        imshow(img)
    plt.show()
    
    
