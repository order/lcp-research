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

DIM = 40

BASIS_TYPE = 'trig'
BASIS_NUM = 2*DIM

PROJ_VAL_REG = 1e-10
PROJ_FLOW_REG = 1e-10
PROJ_ALPHA = 1e-8 # Primal/dual scale term

THRESH = 1e-12
ITER = 2500

#########################################################
# Build objects

def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 10              # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.99           # Discount (\gamma)
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
                                   thresh=1e-10,
                                   max_iter=150)
    print 'Kojima ran for:', time.time() - start, 's'
    P = lcp_builder.expand_block_vector(p,1e-22)
    D = lcp_builder.expand_block_vector(d,1e-22)
    return (P,D,data)

def projective_solve(plcp,lcp_builder):
    q = plcp.q
    Phi = plcp.Phi
    
    (N,k) = Phi.shape
    x0 = np.ones(N)
    y0 = np.ones(N) + q
    w0 = np.zeros(k)
    
    start = time.time()    
    (p,d,data) = solve_with_projective(plcp,
                                       thresh=THRESH,
                                       max_iter=ITER,
                                       x0=x0,
                                       y0=y0,
                                       w0=w0)
    print 'Projective ran for:', time.time() - start, 's'
    
    return (lcp_builder.expand_block_vector(p,1e-22),
            lcp_builder.expand_block_vector(d,1e-22),
            data)

def dumb_projective_solve(plcp,lcp_builder):
    U = plcp.U
    P = plcp.Phi
    Pi = P.dot(P.T)
    q = plcp.q
    
    (N,) = q.shape
    x0 = np.ones(N)
    y0 = np.ones(N)
    w0 = plcp.Phi.T.dot(plcp.q)

    # Explicitly build the projective LCP
    pM = P.dot(U) + (sps.eye(N) - Pi)
    pq = Pi.dot(q)

    dumb_plcp = LCPObj(pM,pq)
    
    start = time.time()    
    (p,d,data) = solve_with_kojima(dumb_plcp,
                                   thresh=THRESH,
                                   max_iter=ITER)
    print 'Dumb way of solving projective ran for:', time.time() - start, 's'
    
    return (lcp_builder.expand_block_vector(p,0),
            lcp_builder.expand_block_vector(d,0),
            data)  

if __name__ == '__main__':

    # Build the MDP and discretizer
    (mdp,disc,_) = build_problem(DIM)

    # Build LCP and builder
    val_reg = 1e-15
    flow_reg = 1e-12
    (lcp,lcp_builder) = build_lcp(mdp,disc,val_reg,flow_reg)
    
    # Solve, initially, using Kojima
    try:
        p = np.load('p.npy')
        d = np.load('d.npy')
        data = {}
        sol = block_solution(mdp,p)
    except Exception:
        (p,d,data) = kojima_solve(lcp,lcp_builder)
        np.save('p.npy',p)
        np.save('d.npy',d)
        sol = block_solution(mdp,p)

        plot_sol_images(mdp,disc,p)
        plt.suptitle('Reference primal')
        plot_sol_images(mdp,disc,d)
        plt.suptitle('Reference dual')
        plt.show()
    # Form the Fourier projection (both value and flow)
    basis = get_basis_from_solution(mdp,disc,sol,BASIS_TYPE,BASIS_NUM)

    (plcp,_) = build_projective_lcp(lcp_builder,
                                    basis,
                                    PROJ_VAL_REG,
                                    PROJ_FLOW_REG,
                                    PROJ_ALPHA)

    (proj_p,proj_d,proj_data) = projective_solve(plcp,lcp_builder)
    #(dumb_p,dumb_d,dumb_data) = dumb_projective_solve(plcp,lcp_builder)
    #print 'Primal error norm', np.linalg.norm(proj_p - dumb_p)
    #print 'Dual error norm', np.linalg.norm(proj_d - dumb_d)
    
    plot_sol_images(mdp,disc,proj_p)
    plt.suptitle('Projected primal')

    plot_sol_images(mdp,disc,proj_d)
    plt.suptitle('Projected dual')
    plt.show()     
