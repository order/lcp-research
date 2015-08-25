import numpy as np
import time
import os
import copy
import math

import scipy.sparse as sps
import scipy
import matplotlib.pyplot as plt
import math
from util import *
import mdp
import lcp

"""
This file contains a number of different iterative solvers for linear complementary problems.
Most are phrased as iteration generators. These plug in to the iter_solver class, which wraps
the generators and provides a standardized framework for recording per-iteration information (like residuals and state) and termination checking (like maximum iteration count or residual threshold checking.
"""

class IterativeSolver(object):

    """
    This is the generic framework for iterative solvers.
    
    Basically does all the scaffolding stuff for the iteration including
    mantaining a list of "recorders" and checking a list of termination conditions.
    """

    def __init__(self,iterator):
        self.recorders = []
        self.termination_conditions = []
        self.iterator = iterator        
        
    def solve(self):
        """
        Call this to use an iteration object to solve an LCP
        
        The iteration solve should have all the LCP information
        Returns a record object.
        """
        assert(len(self.termination_conditions) >= 1)
        
        Records = [[] for _ in xrange(len(self.recorders))]
        
        while True:       
            # First record everything pertinent (record initial information first)
            for (i,recorder) in enumerate(self.recorders):
                Records[i].append(recorder.report(self.iterator))
        
            # Then check for termination conditions
            for term_cond in self.termination_conditions:
                if term_cond.isdone(self.iterator):
                    print 'Termination reason:', term_cond
                    return Records                   
                
            # Finally, advance to the next iteration
            self.iterator.next_iteration()
                
class Iterator(object):
    """
    Abstract definition of an iterator
    """
    def next_iteration(self):
        raise NotImplementedError()
    def get_primal_vector(self):
        raise NotImplementedError()
    def get_dual_vector(self):
        raise NotImplementedError()
    def get_gradient_vector(self):
        raise NotImplementedError()
    def get_iteration(self):
        raise NotImplementedError()
        
class ValueIterator(Iterator):

    def __init__(self,mdp_obj,**kwargs):
        self.mdp = mdp_obj
        self.iteration = 0
        
        n = mdp_obj.num_states
        self.v = kwargs.get('x0',np.ones((n,1)))
        
    def next_iteration(self): 
        """
        Do basic value iteration, but also try to recover the flow variables.
        This is for comparison to MDP-split lcp_obj solving (see mdp_ip_iter)
        """
        
        gamma = self.mdp.discount
        A = self.mdp.num_actions
        
        Vs = []
        for a in xrange(A):     
            P_T = self.mdp.transitions[a].T
            v_a = self.mdp.costs[a] + gamma*P_T.dot(self.v)
            Vs.append(v_a)
        Vs = np.hstack(Vs)
        self.v = np.amin(Vs,axis=1)            
        self.iteration += 1
       
    def get_value_vector(self):
        return self.v
        
    def get_iteration(self):
        return self.iteration
        

def ProjectiveInteriorPointIteration(proj_lcp_obj,state,**kwargs):
    """
    A projective interior point algorithm based on "Fast solutions to projective monotone LCPs" by Geoff Gordon (http://arxiv.org/pdf/1212.6958v1.pdf). If M = \Phi U + \Pi_\bot where Phi is low rank (k << n) and \Pi_\bot is projection onto the nullspace of \Phi^\top, then each iteration only costs O(nk^2), rather than O(n^{2 + \eps}) for a full system solve.

    This is a straight-forward implementation of Figure 2
    """
    DEBUG = False

    """
    RE-CORE this an all other iterative procedures to deal with matrices
    rather than arrays. 
    This leads to better interaction with sparse stuff, and I think is more
    comprehensible to my matlab sensibilities
    """

    sigma = kwargs.get('centering_coeff',0.99)
    beta = kwargs.get('linesearch_backoff',0.99)
    backoff = kwargs.get('central_path_backoff',0.9995)
 
    # Currently: converting everything to a dense matrix
    if hasattr(proj_lcp_obj,'todense'):
        Phi = np.array(proj_lcp_obj.Phi.todense())
        U = np.array(proj_lcp_obj.U.todense())
        q = np.array(proj_lcp_obj.q.todense())
    else:
        Phi = proj_lcp_obj.Phi
        U = proj_lcp_obj.U
        q = proj_lcp_obj.q
    (N,k) = Phi.shape

    # Preprocess
    PtP = (Phi.T).dot(Phi) # FTF in Geoff's code
    PtPUP = (PtP.dot(U)).dot(Phi)
    PtPU_P = (PtP.dot(U) - Phi.T) # FMI in Geoff's code; I sometimes call it Psi in my notes  

    debug_mapprint(DEBUG,PtP=PtP.shape,PtPU_P=PtPU_P.shape)
    
    
    x = np.ones(N)
    y = np.ones(N)
    w = np.zeros(k)
    
    I = 0
    while True:
        I += 1
        debug_print(DEBUG, '-'*5+str(I)+'-'*5)
        debug_mapprint(DEBUG,x=x,y=y,w=w)

        # Step 3: form right-hand sides
        g = sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        #pinv_phi_x = scipy.linalg.lstsq(Phi,x)[0] # TODO: use QR factorization
        #r = Phi * (U * x - pinv_phi_x) + x + q - y # M*x + q - y
        p = x - y + q - Phi.dot(w)
        # TODO: Geoff doesn't explicitly form r (rhs2 in his code), figure this out
        debug_mapprint(DEBUG,g=g,p=p)

        # Step 4: Form the reduced system G dw = h
        inv_XY = 1.0/(x + y)
        A = PtPU_P * inv_XY
        G = (A * y).dot(Phi) - PtPUP
        h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
        debug_mapprint(DEBUG,h=h,G=G)
        
        # Step 5: Solve for del_w

        del_w = np.linalg.lstsq(G,h)[0]
        
        # Step 6
        Phidw= Phi.dot(del_w)
        del_y = inv_XY*(g + y*p - y*Phidw)
        
        # Step 7
        del_x = del_y+Phidw-p
        
        debug_mapprint(DEBUG,del_x=del_x,del_y=del_y,del_w=del_w)
        
        # Step 8 Step length
        steplen = max(np.amax(-del_x/x),np.amax(-del_y/y))
        debug_mapprint(DEBUG,steplen_pre=steplen,beta=sigma)
        if steplen <= 0:
            steplen = float('inf')
        else:
            steplen = 1.0 / steplen
        steplen = min(1.0, 0.666*steplen + (backoff - 0.666) * steplen**2)
                
        # Sigma is beta in Geoff's code
        if(steplen > 0.95):
            sigma = 0.05 # Long step
        else:
            sigma = 0.5 # Short step
        debug_mapprint(DEBUG,steplen=steplen,beta=sigma)
           
        
        
        x = x + steplen * del_x
        y = y + steplen * del_y
        w = w + steplen * del_w
        
        state.x = x
        state.w = y # Yeah, this is confusing. Should use a different name for this like dual
        state.iter = I
        
        yield state
    
        
def mdp_ip_iter(lcp_obj,state,**kwargs):
    """
    Splits an lcp_obj based on an MDP into (B,C) blocks that correspond to
    value iteration. Cottle 5.2 has more details on general splitting
    methods like PSOR
    """
    assert('MDP' in kwargs)
    MDP = kwargs['MDP'] # Must be present; don't want to do size inference

    M = lcp_obj.M
    q = lcp_obj.q
    sigma = kwargs.get('centering_coeff',0.1)
    beta = kwargs.get('linesearch_backoff',0.8)
    
    solve_thresh = kwargs.get('mdp_split_inner_thresh',1e-6)
    
    N = lcp_obj.dim
    x = np.ones(N)
    y = np.ones(N)
    
    split = mdp.MDPValueIterSplitter(MDP) # Splitter based on value iter

    Total_I = 0
    Outer_I = 0
    while True:
        # Update q split based on current x
        (B,q_k) = split.update(x) 
        # Use Kojima's UIP to solve lcp_obj(B,q_k)
        inner_solver = kojima_ip_iter(lcp.LCPObj(B,q_k),state,**kwargs)
        Inner_I = 0        
        
        # Want both complementarity and residual to be small before
        # stopping the inner iter
        while x.dot(y) >= solve_thresh\
            or np.linalg.norm(y - (q_k + B.dot(x))) > solve_thresh:
            state = inner_solver.next()
            x = state.x
            y = state.w
            Inner_I += 1
            Total_I += 1
        
        #Use the actual Mx+q rather than IP solver's 
        state.w = M.dot(x) + q
        state.iter = Outer_I # Could use other counters
        yield state  
        Outer_I += 1
        

##############################
# Kojima lcp_obj

def kojima_ip_iter(lcp_obj,state,**kwargs):
    """Interior point solver based on Kojima et al's
    "A Unified Approach to Interior Point Algorithms for lcp_objs"
    Uses a log-barrier w/centering parameter
    (How is this different than the basic scheme a la Nocedal and Wright?)
    """
    M = lcp_obj.M
    q = lcp_obj.q

    sigma = kwargs.get('centering_coeff',0.1)
    beta = kwargs.get('linesearch_backoff',0.8)   
    
    N = lcp_obj.dim
    x = np.ones(N)
    y = np.ones(N)
    
    dot = float('inf')
    I = 0
    Bottom = sps.hstack([-sps.csr_matrix(M),sps.eye(N)])
    X = sps.diags(x,0)
    Y = sps.diags(y,0)
    Top = sps.hstack((Y, X))
    A = sps.vstack([Top,Bottom]).tocsr()

    while True:
        assert(not np.any(x < 0))
        assert(not np.any(y < 0))
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)
        
        # Set up Newton direction equations         
        b = np.concatenate([sigma * dot / float(N) * np.ones(N) - x*y, r])
        X = sps.diags(x,0)
        Y = sps.diags(y,0)
        Top = sps.hstack((Y, X))
        A = sps.vstack([Top,Bottom]).tocsr()
            
        # Solve Newton direction equations
        #start = time.time()
        assert(sps.issparse(A))
        dir = sps.linalg.spsolve(A,b)
        #print '\tSparse solve time:', time.time() - start 
        dir_x = dir[:N]
        dir_y = dir[N:]
        
        # Get step length so that x,y are strictly feasible after move
        step = 1
        x_cand = x + step*dir_x
        y_cand = y + step*dir_y    
        while np.any(x_cand < 0) or np.any(y_cand < 0):
            step *= beta
            x_cand = x + step*dir_x
            y_cand = y + step*dir_y

        x = x_cand
        y = y_cand
        
        state.x = x
        state.w = y
        state.iter = I
        I += 1
        yield state
    

##############################
# Basic infeasible-start interior point (p.159 / pdf 180 Wright)
def basic_ip_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    k = 0
    Top = sps.hstack([M,-sps.eye(M.shape[0])])
    x = np.ones(N)
    s = np.ones(N)
    sigma = kwargs.get('centering_coeff',0.01)
    beta = kwargs.get('linesearch_backoff',0.9)
    
    while True:
        assert(not any(x < 0))
        assert(not any(s < 0))
        
        mu = x.dot(s) / N # Duality measure
        
        # Construct matrix from invariant Top and diagonal bottom
        X = sps.diags(x,0)
        S = sps.diags(s,0)
        Bottom = sps.hstack([S, X])
        A = sps.vstack([Top,Bottom]).tocsr()
        r = s - M.dot(x) - q
        centering = -x*s + sigma*mu*np.ones(N)

        b = np.hstack([r, centering])
        
        # Solve, break solution into two parts
        d = sps.linalg.spsolve(A,b)        
        dx = d[:N]
        ds = d[N:]
        
        # Backtrack to maintain positivity.
        alpha = 1
        x_cand = x + alpha*dx
        s_cand = s + alpha*ds
        while any(x_cand < 0) or any (s_cand < 0):
            alpha *= beta
            x_cand = x + alpha*dx
            s_cand = s + alpha*ds
            
        x = x_cand
        s = s_cand
        state.x = x
        state.w = s
        state.iter = k
        k += 1
        yield state


  
##############################
# Iteration generators
    
def euler_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step = kwargs['step']
    
    x = state.x
    w = M.dot(x) + q
    while True:
        x = nonneg_proj(x - step * w)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
        
def euler_barrier_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step = kwargs['step']
    
    wolfe_const = kwargs.get('wolfe_const',1e-4)
    step_decay = kwargs.get('step_decay',0.9)
    min_step = kwargs.get('min_step', 1e-4)
        
    x = state.x
    x[x < 1e-9] = 1e-9
    
    c = 0.25
    
    w = M.dot(x) + q + c/x
    I = 1.0
    while True:
        step = 1
        c *= 0.95

        fx = fb_residual(x,w)
        grad_norm = np.linalg.norm(w)
        while True:
            x_cand = nonneg_proj(x - step * w)
            w_cand = M.dot(x_cand) + q + c/x_cand
            fx_cand = fb_residual(x_cand,w_cand) # Min the Fischer-Burmeister residual
            if fx_cand <= fx - wolfe_const*step*grad_norm**2:
                break
            if step <= min_step:
                step = min_step
                break
            step *= step_decay
        w = w_cand
        x = x_cand
        
        assert(isvector(x))
        assert(x.size == N)
        
        I += 1.0
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def euler_linesearch_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    wolfe_const = kwargs.get('wolfe_const',1e-4)
    step_decay = kwargs.get('step_decay',0.9)
    min_step = kwargs.get('min_step', 1e-4)
    N = lcp_obj.dim
    
    x = state.x
    w = M.dot(x) + q
    while True:
        step = 1
        fx = fb_residual(x,w)
        grad_norm = np.linalg.norm(w)
        while True:
            x_cand = nonneg_proj(x - step * w)
            w_cand = M.dot(x_cand) + q
            fx_cand = fb_residual(x_cand,w_cand) # Min the Fischer-Burmeister residual
            if fx_cand <= fx - wolfe_const*step*grad_norm**2:
                break
            if step <= min_step:
                step = min_step
                break
            step *= step_decay
        w = w_cand
        x = x_cand
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state 
        
def euler_speedy(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step_base = kwargs['step']
    
    Momentum = 0
    IncToken = 1
    DecelToken = 8
    BreakToken = 135
    IncThresh = 0.997
    BreakThresh = 0.97
    AngleEps = 7e-6
    ScoreEps = 5e-2
    Multiplier = 1.03
    
    x = state.x
    x_old = x
    w = M.dot(x) + q
    w = w / np.linalg.norm(w)
    w_old = w
    theta_old = 0
    
    score_best = float('inf');
    score_old = float('inf');
    while True:
        theta = w_old.dot(w)
        if theta > IncThresh and theta >= (1.0 - AngleEps)*theta_old:
            Momentum += IncToken
        if theta <= BreakThresh:
            Momentum -= BreakToken
            x = x_old
            w = M.dot(x) + q
            w = w / np.linalg.norm(w)
            
        step = step_base * Multiplier**Momentum
        x_old = x
        w_old = w
        theta_old = theta
        
        x = nonneg_proj(x - step * w)
        w = M.dot(x) + q
        w = w / np.linalg.norm(w)
        
        assert(isvector(x))
        assert(x.size == N)
        
        score = basic_residual(x,w)
        if score <= score_best:
            score_best = score
        if score > (1+ScoreEps)*score_old:
            Momentum -= DecelToken 
        print Momentum
            
        state.iter += 1
        state.x = x
        state.w = w
        yield state   
    
def projected_jacobi_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    assert(has_pos_diag(M))
    N = lcp_obj.dim
    D_inv = np.diag(np.diag(M))
    scale = kwargs.get('scale',1.0);    
    x = state.x
    w = M.dot(x) + q
    while True:
        x = nonneg_proj(x - scale*D_inv.dot(w))
        w = M.dot(x) + q

        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def psor_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q
    assert(has_pos_diag(M))
    N = lcp_obj.dim
    relax = kwargs.get('omega',1.0)
    
    x = state.x
    while True:
        x = proj_forward_prop(x,M,q,relax)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
def extragrad_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q
    N = lcp_obj.dim
    step = kwargs['step']
    
    x = state.x
    w = M.dot(x) + q
    while True:
        y = nonneg_proj(x - step * w)
        v = M.dot(y) + q
        x = nonneg_proj(x - step * v)
        w = M.dot(x) + q
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = w
        yield state
        
		
# Based on "Adaptive Restart for Accelerated Gradient Schemes" by O'Donoghue and Candes
# http://arxiv.org/pdf/1204.3982v1.pdf
def accelerated_prox_iter(lcp_obj,state,**kwargs):
    M = lcp_obj.M
    q = lcp_obj.q

    N = lcp_obj.dim
    step = kwargs['step']
    restart = kwargs.get('restart',0.1)
    
    x = state.x
    w = M.dot(x) + q
    k = 1
    state.x_prev = x 
    y = x
    theta = 1
    
    while True:
        grad = M.dot(y) + q # grad f(y^k)        
        state.x_prev = x
        x = nonneg_proj(y - step * grad)
        theta_old = theta
        theta = quad(1,theta**2 - restart,-theta**2)[0]
        beta = theta_old * (1 - theta_old) / (theta_old**2 + theta)
        y = x + beta * (x - state.x_prev)
        
        if grad.dot(x - state.x_prev) > 0:
            theta = 1
            y = x
            print state.iter
        
        
        assert(isvector(x))
        assert(x.size == N)
        
        state.iter += 1
        state.x = x
        state.w = grad
        yield state
        
        
