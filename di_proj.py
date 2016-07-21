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
from linalg import hstack

from utils import *

from experiment import *

DIM = 32

BASIS_TYPE = 'jigsaw'
BASIS_NUM = 10

REGEN = True

REG = 1e-6
VAL_REG = REG
FLOW_REG = REG
PROJ_VAL_REG = REG
PROJ_FLOW_REG = REG
PROJ_ALPHA = 1e-6

THRESH = 1e-20
ITER = 500

PROJ_THRESH = 1e-20
PROJ_ITER = 2500

#########################################################
# Build objects

def build_problem(disc_n):
    # disc_n = number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 10              # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.95            # Discount (\gamma)
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
    #lcp_builder.add_drain(np.zeros(2),0.0)
    
    lcp_builder.build_uniform_state_weights()
    
    lcp = lcp_builder.build()

    M = lcp.M
    return (lcp,lcp_builder)

def build_projective_lcp(lcp_builder,basis,val_reg,flow_reg,scale):
    lcp_builder.val_reg = val_reg
    lcp_builder.flow_reg = flow_reg

    lcp = lcp_builder.build()
  
    (N,n) = lcp.M.shape
    assert(N == n)
    
    start = time.time()
    # Remove omitted nodes
    
    #P = lcp_builder.contract_block_matrix(basis.toarray())
    #P = linalg.orthonorm(P)
    P = basis
    (n,k) = P.shape
    # Right size
    assert(k <= n)
    assert(N == n)
    #Basicaly orthonormal
    assert((P.T.dot(P) - sps.eye(k)).sum()/float(k) < 1e-12)
    
    # Convert to sparse and elim zeros
    P = sps.csr_matrix(P)
    P.eliminate_zeros()
    M = lcp.M.tocsr()
    M.eliminate_zeros()

    assert(N == P.shape[0])    
    # Project MDP onto basis
    U = scale * P.T.dot(M) # Assumes that B is orthonormal
    PtPU = U # Also assumes B is orthonormal

    q = scale * P.dot(P.T.dot(lcp.q))
    
    plcp = ProjectiveLCPObj(P,U,PtPU,q)

    M = plcp.form_M()
    print 'Building projective LCP: {0}s'.format(time.time() - start)
    return (plcp,P)

###############################################
# Solvers

def kojima_solve(lcp,**kwargs):
    # Solve
    (N,) = lcp.q.shape
    x0 = kwargs.get('x0',np.ones(N))
    y0 = kwargs.get('y0',np.ones(N))
    start = time.time()
    (p,d,data) = solve_with_kojima(lcp,
                                   thresh=THRESH,
                                   max_iter=ITER,
                                   x0=x0,
                                   y0=y0)
    print 'Kojima ran for:', time.time() - start, 's'
    return (p,d,data)

def projective_solve(plcp,x0,y0,w0):        
    start = time.time()    
    (p,d,data) = solve_with_projective(plcp,
                                       thresh=PROJ_THRESH,
                                       max_iter=PROJ_ITER,
                                       x0=x0,
                                       y0=y0,
                                       w0=w0)
    print 'Projective ran for:', time.time() - start, 's'
    return (p,d,data)

def dumb_projective_solve(plcp,lcp_builder,x0,y0,w0):
    # Explicitly build the projective LCP
    q = plcp.q
    (N,) = q.shape
    M = plcp.form_M()

    dumb_plcp = LCPObj(M,q)
    
    start = time.time()    
    (p,d,data) = solve_with_kojima(dumb_plcp,
                                   thresh=PROJ_THRESH,
                                   max_iter=PROJ_ITER,
                                   x0=x0,
                                   y0=y0)
    print 'Dumb way of solving projective ran for:', time.time() - start, 's'
    
    return (p,d,data)

if __name__ == '__main__':

    # Build the MDP and discretizer
    (mdp,disc,_) = build_problem(DIM)

    # Build LCP and builder
    (lcp,lcp_builder) = build_lcp(mdp,disc,VAL_REG,FLOW_REG)
    
    # Solve, initially, using Kojima
    try:
        assert(not REGEN)
        p = np.load('p.npy')
        d = np.load('d.npy')
        data = {}
        p_sol = block_solution(mdp,p)
        d_sol = block_solution(mdp,d)
        
    except Exception:
        #(x0,y0,z) = generate_initial_feasible_points(lcp.M,
        #                                             lcp.q)

        # Create an augmented LCP system with known
        # Feasible start
        alcp,x0,y0 = augment_lcp(lcp,1e3)
        (N,) = alcp.q.shape
        assert(N == lcp.q.shape[0] + 1) # Augmented
        
        (p,d,data) = kojima_solve(alcp,
                                  x0=np.ones(N),
                                  y0=np.ones(N))
        print 'Slack variables', p[-1],d[-1]

        # Strip augmented variable and expand omitted nodes
        p = p[:-1]
        d = d[:-1]
        p = lcp_builder.expand_block_vector(p,1e-22)
        d = lcp_builder.expand_block_vector(d,1e-22)
        
        np.save('p.npy',p)
        np.save('d.npy',d)

        if False:
            plot_sol_images_interp(mdp,disc,p)
            plt.suptitle('Reference primal')
            plot_sol_images_interp(mdp,disc,d)
            plt.suptitle('Reference dual')
            plt.show()
        
        p_sol = block_solution(mdp,p)
        d_sol = block_solution(mdp,d)
    
    # Form the Fourier projection (both value and flow)
    M = lcp.M
    q = lcp.q
    contract_p = lcp_builder.contract_block_vector(p)
    contract_d = lcp_builder.contract_block_vector(d)
    assert(q.shape == contract_p.shape)
    assert(q.shape == contract_d.shape)

    recon_d = M.dot(contract_p) + q
    print 'Inner product:', contract_p.dot(contract_d)
    print 'Dual reconstruction error', np.linalg.norm(contract_d - recon_d)
    
    """
    basis = get_basis_from_solution(mdp,
                                    disc,
                                    p_sol,
                                    d_sol,
                                    BASIS_TYPE,
                                    BASIS_NUM)
    """
    # Value basis; approximate v, E_1f_1, E_2F_2,p
    E1 = mdp.get_E_matrix(0)
    E1 = lcp_builder.contract_sparse_square_matrix(E1)
    E2 = mdp.get_E_matrix(1)
    E2 = lcp_builder.contract_sparse_square_matrix(E2)

    #print np.linalg.cond(E1.toarray())
    #print np.linalg.cond(E2.toarray())

    c = lcp_builder.contract_vector(mdp.costs[0])
    
    weights = np.ones(c.size) / float(c.size)
    contract_p_sol = lcp_builder.contract_solution_block(p_sol)
    contract_d_sol = lcp_builder.contract_solution_block(d_sol)

    [v,f1,f2] = [contract_p_sol[:,i] for i in range(3)]
    [u,g1,g2] = [contract_d_sol[:,i] for i in range(3)]

    ##
    s = 1
    v = v*np.abs(0.01*np.random.randn(v.size)+1)
    #f1 *= (1 + s*(2*np.random.rand(v.size)-1))
    #f2 *= (1 + s*(2*np.random.rand(v.size)-1))
    Bv = hstack([v,
                 #E1.dot(f1),
                 #E2.dot(f2),
                 np.random.rand(v.size,10),
                 weights])
    Bf1 = hstack([f1,
                  E1.T.dot(v),
                  c,
                  np.random.rand(v.size,27)])
    Bf1 = np.eye(v.size)
    Bf2 = hstack([f2,
                  E2.T.dot(v),
                  c,
                  np.random.rand(v.size,27)])
    Bf2 = np.eye(v.size)
    
    Bv = orthonorm(Bv)
    Bf1 = orthonorm(Bf1)
    Bf2 = orthonorm(Bf2)    
    B = sps.block_diag([Bv,Bf1,Bf2])

    proj_p = B.dot(B.T.dot(contract_p))
    proj_d = B.dot(B.T.dot(contract_d))

    p_res = contract_p - proj_p
    d_res = contract_d - proj_d
    print 'Primal projection residual', np.linalg.norm(p_res)
    print 'Dual projection residual', np.linalg.norm(d_res)

    (plcp,_) = build_projective_lcp(lcp_builder,
                                    B,
                                    PROJ_VAL_REG,
                                    PROJ_FLOW_REG,
                                    PROJ_ALPHA)

    # Generate feasible initial points
    x = lcp_builder.contract_block_vector(p)
    y = lcp_builder.contract_block_vector(d)
    x = np.ones(x.size)
    y = np.ones(y.size)
    aplcp,x0,y0,w0 = augment_plcp(plcp,1e2,x0=x,y0=y)
    (proj_p,proj_d,proj_data) = projective_solve(aplcp,
                                                 x0,y0,w0)

    
    print 'Projected infeasibility:', proj_p[-1]
    # Strip augmented variable and expand omitted nodes
    proj_p = proj_p[:-1]
    proj_d = proj_d[:-1]
    proj_p = lcp_builder.expand_block_vector(proj_p,1e-35)
    proj_d = lcp_builder.expand_block_vector(proj_d,1e-35)
    #plot_data_dict(proj_data)
    
    plot_sol_images_interp(mdp,disc,proj_p)
    plt.suptitle('Projected primal')

    plot_sol_images_interp(mdp,disc,proj_d)
    plt.suptitle('Projected dual')
    plt.show()     
