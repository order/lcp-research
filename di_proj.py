import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.linalg as spl
import multiprocessing

import matplotlib.pyplot as plt

from collections import defaultdict

from mdp import *
from config.mdp import *
from config.solver import *
from utils import *
from discrete import *
from lcp import *

from utils import *

from experiment import *

DIM = 32

REBUILD = True

#########################################################
# Build objects

def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 1               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 5.0               # Control jitter 
    discount = 0.995          # Discount (\gamma)
    B = 5
    bounds = [[-B,B],[-B,B]]  # Square bounds, 
    cost_radius = 0.25        # Goal region radius
    
    actions = np.array([[-1],[1]]) # Actions
    action_n = actions.shape[0]
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

def build_lcp(mdp,disc,val_reg,flow_reg):
    lcp_builder = LCPBuilder(mdp,
                             disc,
                             val_reg=val_reg,
                             flow_reg=flow_reg)

    lcp_builder.remove_oobs(0.0)
    lcp_builder.add_drain(np.zeros(2),0.0)
    
    lcp_builder.build_uniform_state_weights()
    
    lcp = lcp_builder.build()
    return(lcp,lcp_builder)

def build_projective_lcp(lcp_builder,basis,val_reg,flow_reg,scale):
    lcp_builder.val_reg = val_reg
    lcp_builder.flow_reg = flow_reg

    lcp = lcp_builder.build()
  
    (N,n) = lcp.M.shape
    assert(N == n)
    
    start = time.time()
    # Remove omitted nodes
    
    B = lcp_builder.contract_block_matrix(basis.toarray())

    B = linalg.orthonorm(B)
    # Convert to sparse and elim zeros
    B = sps.csr_matrix(B)
    B.eliminate_zeros()
    M = lcp.M.tocsr()
    M.eliminate_zeros()

    assert(N == B.shape[0])    
    # Project MDP onto basis
    U = B.T.dot(M) # Assumes that B is orthonormal
    
    PtPU = U # Also assumes B is orthonormal
    plcp = ProjectiveLCPObj(B,scale*U,scale*PtPU, scale*lcp.q)
    print 'Building projective LCP: {0}s'.format(time.time() - start)
    return (plcp,B)

###############################################
# Solvers

def kojima_solve(lcp,lcp_builder):
    # Solve
    start = time.time()
    (p,d,data) = solve_with_kojima(lcp,
                                   thresh=1e-12,
                                   max_iter=150)
    print 'Kojima ran for:', time.time() - start, 's'
    P = lcp_builder.expand_block_vector(p,1e-22)
    D = lcp_builder.expand_block_vector(d,1e-22)
    return (P,D,data)

def projective_solve(plcp,p,d):
    start = time.time()
    (N,k) = plcp.Phi.shape

    print 'q', plcp.q.shape
    print 'PtPU', plcp.PtPU.shape
    print 'Phi', plcp.Phi.shape
    x0 = p
    y0 = d

    w0 = plcp.Phi.T.dot(x0 - y0 + plcp.q)
    assert((k,) == w0.shape)

        
    (p,d,data) = solve_with_projective(plcp,
                                       thresh=1e-12,
                                       max_iter=250,
                                       x0=x0,
                                       y0=y0,
                                       w0=w0)
    print 'Projective ran for:', time.time() - start, 's'
    return (p,d,data)

def projective_regularization_homotopy(mdp,disc,basis):
    
    (lcp,lcp_builder) = build_lcp(mdp,disc,0,0)
    (N,) = lcp.q.shape
    p = np.ones(N)
    d = np.ones(N)

    val_reg = 1e-6
    flow_reg = 1e-6
    
    (plcp,_) = build_projective_lcp(lcp_builder,
                                    basis,
                                    val_reg,
                                    flow_reg,
                                    1) # Scale term
    (p,d,data) = projective_solve(plcp,p,d)

    if False:
        # Movie
        X = np.array(data['x']).T
        frames = lcp_internal_to_frames(mdp,disc,lcp_builder,X,0)
        animate_frames(frames)
        quit()
    
    if True:
        # Trajectory plots
        plot_data_dict(data)
            
    return (lcp_builder.expand_block_vector(p),
            lcp_builder.expand_block_vector(d))
            

if __name__ == '__main__':

    ####################################################
    # Build the MDP and discretizer
    (mdp,disc,_) = build_problem(DIM)

    ####################################################
    # Build LCP and builder
    val_reg = 1e-15
    flow_reg = 1e-12
    (lcp,lcp_builder) = build_lcp(mdp,disc,val_reg,flow_reg)
    
    ####################################################
    # Solve, initially, using Kojima
        # Build / load
    if REBUILD:
        (p,d,data) = kojima_solve(lcp,lcp_builder)
        np.save('p.npy',p)
        np.save('d.npy',d)
        
    else:
        p = np.load('p.npy')
        d = np.load('d.npy')
        data = {}
    sol = block_solution(mdp,p)
    dsol = block_solution(mdp,d)

    if REBUILD and False:
        plot_data_dict(data)
    
    A = mdp.num_actions+1

    if False:
        # CDF plots for final primal / dual
        (x,F) = cdf_points(p * d)
        (y,G) = cdf_points(p + d)
        plt.figure()
        plt.suptitle('CDFs')
        plt.subplot(1,2,1);
        plt.plot(x,F)
        plt.title('P*D')
        plt.subplot(1,2,2)
        plt.semilogx(y,G)
        plt.title('P+D')

    if True:
        # Image plots for final primal / dual
        plt.figure()
        plt.suptitle('Primal')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(sol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()
        
        plt.figure()
        plt.suptitle('Dual')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(dsol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()
        
        plt.figure()
        plt.suptitle('Primal + Dual')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(sol[:,i] + dsol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()

        plt.figure()
        plt.suptitle('Primal * Dual')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(sol[:,i] * dsol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()
        plt.show()

    ####################################################
    # Form the Fourier projection (both value and flow)
    basis = get_basis_from_solution(mdp,
                                    disc,
                                    sol,
                                    'trig',
                                    150)
    proj_p,proj_d = projective_regularization_homotopy(mdp,disc,basis)

    proj_sol = block_solution(mdp,proj_p)
    proj_dsol = block_solution(mdp,proj_d)


    if True:
        # Show images of the projected solution
        plt.figure()
        plt.suptitle('Projected primal')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(proj_sol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()

        plt.figure()
        plt.suptitle('Projected Dual')
        for i in xrange(A):
            plt.subplot(2,2,i+1)
            img = reshape_full(proj_dsol[:,i],disc)
            plt.imshow(img,interpolation='none')
            plt.colorbar()
    plt.show()
        
