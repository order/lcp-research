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

DIM = 16

def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 1               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1             # Control jitter 
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

def build_projective_lcp(mdp,disc,basis):
    n = mdp.num_states
    A = mdp.num_actions

    unreach = find_isolated(mdp,disc)
    index_mask = np.ones(n)
    index_mask[unreach] = 0
    indices = np.where(index_mask == 1.0)[0]
    I = indices.size
    
    start = time.time()
    # Build the LCP
    lcp = mdp.build_lcp(indices=indices,
                        val_reg = 1e-6,
                        flow_reg = 1e-6)

    print 'LCP shape:',lcp.M.shape
    print 'Anticipated:',(n-unreach.size)*(A+1)

    # TODO: indices need to be repeated "per block"
    B = linalg.orthonorm(basis.toarray()[indices,:])
    assert(I*(A+1) == B.shape[0])
    
    # Convert matrices to sparse and elim zeros
    B = sps.csr_matrix(B)
    B.eliminate_zeros()
    M = linalg.spsubmat(lcp.M,indices,indices)
    M.eliminate_zeros()

    # Project MDP onto basis
    PtPU = B.T.dot(M)
    plcp = ProjectiveLCPObj(B, PtPU, lcp.q[indices])    
    print 'Building projective LCP: {0}s'.format(time.time() - start)
    return (plcp,indices)

def solve_mdp_with_projective(mdp,disc,basis,p,d):
    (plcp,included_states) = build_projective_lcp(mdp,disc,basis)
    start = time.time()
    (N,k) = plcp.Phi.shape

    print 'q', plcp.q.shape
    print 'PtPU', plcp.PtPU.shape
    print 'Phi', plcp.Phi.shape
    x0 = (1e-6)*np.ones(N)    
    y0 = np.maximum(plcp.q,1e-6)
    w0 = np.zeros(k)

        
    (p,d) = solve_with_projective(plcp,1e-12,100,x0,y0,w0)
    print 'Projective ran for:', time.time() - start, 's'
    (p,d) = expand_states(mdp,p,d,included_states)
    return block_solution(mdp,p)

if __name__ == '__main__':

    ####################################################
    # Build the MDP and discretizer
    (mdp,disc,_) = build_problem(DIM)
    
    ####################################################
    # Solve, initially, using Kojima
        # Build / load
    if False:
        (p,d) = solve_mdp_with_kojima(mdp)
        np.save('p.npy',p)
        np.save('d.npy',d)
    else:
        p = np.load('p.npy')
        d = np.load('d.npy')         
    sol = block_solution(mdp,p)

    ####################################################
    # Only use nodes that are reachable
    unreach = find_isolated(mdp,disc)
    index_mask = np.ones(mdp.num_states)
    index_mask[unreach] = 0
    indices = np.where(index_mask == 1.0)[0]
    # These are the indicies of active nodes

    ####################################################
    # Form the Fourier projection (both value and flow)
    basis = get_basis_from_solution(mdp,disc,indices,sol,100)
    
    #B = np.eye(p.size)
    print 'Basis shape:',basis.shape
    # Solve with projective method
    start = time.time()
    p_sol = solve_mdp_with_projective(mdp,disc,basis,p,d)
    ptime = time.time() - start
    for i in xrange(4):
        plt.subplot(2,2,i+1)
        img = reshape_full(p_sol[:,i],disc)
        imshow(img)
    plt.show()
    
    
