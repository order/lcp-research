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
        self.iter_message = None
        
    def solve(self):
        """
        Call this to use an iteration object to solve an LCP
        
        The iteration solve should have all the LCP information
        Returns a record object.
        """
        assert(len(self.termination_conditions) >= 1)
        
        Records = [[] for _ in xrange(len(self.recorders))]
        
        while True:       
            if self.iter_message:
                print self.iter_message
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
    def get_iteration(self):
        raise NotImplementedError()
        
class LCPIterator(Iterator):
    def get_primal_vector(self):
        raise NotImplementedError()
    def get_dual_vector(self):
        raise NotImplementedError()
    def get_gradient_vector(self):
        raise NotImplementedError()

        
class MDPIterator(Iterator):
    def get_value_vector(self):
        raise NotImplementedError()
        
class ValueIterator(MDPIterator):
    def __init__(self,mdp_obj,**kwargs):
        self.mdp = mdp_obj
        self.iteration = 0
        
        n = mdp_obj.num_states
        self.v = kwargs.get('v0',np.ones(n))
        
    def next_iteration(self): 
        """
        Do basic value iteration, but also try to recover the flow variables.
        This is for comparison to MDP-split lcp_obj solving (see mdp_ip_iter)
        """
        
        gamma = self.mdp.discount
        A = self.mdp.num_actions
        N = self.mdp.num_states
        
        Vs = np.zeros((N,A))
        for a in xrange(A):     
            P_T = self.mdp.transitions[a].T
            Vs[:,a] = self.mdp.costs[a] + gamma*P_T.dot(self.v)
            assert(not np.any(np.isnan(Vs[:,a])))
            
        self.v = np.amin(Vs,axis=1)
        self.iteration += 1
       
    def get_value_vector(self):
        return self.v
        
    def get_iteration(self):
        return self.iteration
        
##############################
# Kojima lcp_obj
class KojimaIterator(LCPIterator):
    def __init__(self,lcp_obj,**kwargs):
        self.lcp = lcp_obj
        self.M = lcp_obj.M
        self.q = lcp_obj.q
        
        self.centering_coeff = kwargs.get('centering_coeff',0.1)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',0.8)  
        
        self.iteration = 0
        
        n = lcp_obj.dim
        
        self.x = kwargs.get('x0',np.ones(n))
        self.y = kwargs.get('y0',np.ones(n))
        assert(not np.any(self.x < 0))
        assert(not np.any(self.y < 0))
        
    def next_iteration(self):
        """Interior point solver based on Kojima et al's
        "A Unified Approach to Interior Point Algorithms for lcp_objs"
        Uses a log-barrier w/centering parameter
        (How is this different than the basic scheme a la Nocedal and Wright?)
        """
        M = self.lcp.M
        q = self.lcp.q
        n = self.lcp.dim
        
        x = self.x
        y = self.y

        sigma = self.centering_coeff
        beta = self.linesearch_backoff
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)
            
            # Set up Newton direction equations
        A = sps.bmat([[sps.spdiags(y,0,n,n),sps.spdiags(x,0,n,n)],\
            [-sps.coo_matrix(M),sps.eye(n)]],format='csr')          
        b = np.concatenate([sigma * dot / float(n) * np.ones(n) - x*y, r])
                
        dir = sps.linalg.spsolve(A,b)
        dir_x = dir[:n]
        dir_y = dir[n:]
            
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
        
        self.x = x
        self.y = y
        self.iteration += 1
        
    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
        

class ProjectiveIteration(LCPIterator):
    def __init__(self,proj_lcp_obj,**kwargs):
        self.centering_coeff = kwargs.get('centering_coeff',0.99)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',0.99)
        self.central_path_backoff = kwargs.get('central_path_backoff',0.9995)
        
        self.proj_lcp_obj = proj_lcp_obj
        self.q = proj_lcp_obj.q
        self.Phi = proj_lcp_obj.Phi
        self.U = proj_lcp_obj.U
        
        # We expect Phi and U to be both either dense, or both sparse (no mixing)
        # If sparse, we want them to be csr or csc (these can be mixed)
        # self.linsolve is set based on whether we are in the sparse regime or the dense one.
        # TODO: also allow mixed sparseness (e.g. Phi dense, U csr)
        if isinstance(self.Phi,np.ndarray):
            assert(isinstance(self.U,np.ndarray))
            self.linsolve = lambda A,b: np.linalg.lstsq(G,h)[0]
        else:
            assert(isinstance(self.Phi,sps.csc_matrix) or isinstance(self.Phi,sps.csr_matrix))
            assert(isinstance(self.U,sps.csc_matrix) or isinstance(self.U,sps.csr_matrix))
            self.linsolve = lambda A,b: sps.linalg.lsqr(G,h)[0]
         
        # Preprocess
        self.PtP = (Phi.T).dot(Phi) # FTF in Geoff's code
        self.PtPUP = (PtP.dot(U)).dot(Phi)
        self.PtPU_P = (PtP.dot(U) - Phi.T) # FMI in Geoff's code; I sometimes call it Psi in my notes
        
        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(k))

        self.iteration = 0
        
    def next_iteration(self):
        """
        A projective interior point algorithm based on "Fast solutions to projective monotone LCPs" by Geoff Gordon. 
        If M = \Phi U + \Pi_\bot where Phi is low rank (k << n) and \Pi_\bot is projection onto the nullspace of \Phi^\top, then each iteration only costs O(nk^2), rather than O(n^{2 + \eps}) for a full system solve.

        This is a straight-forward implementation of Figure 2
        """

        sigma = self.centering_coeff
        beta = self.linesearch_backoff
        backoff = self.central_path_backoff
     
        Phi = self.Phi
        U = self.U
        q = self.q
        assert(len(Phi.shape) == 2)
        (N,k) = Phi.shape
        assert(ismatrix(U,k,N))
        assert(isvector(q,N))

        PtP = self.PtP # FTF in Geoff's code
        assert(issquare(PtP,k))
        PtPUP = self.PtPUP
        assert(issquare(PtPUP,k))
        PtPU_P = self.PtPU_P 
        assert(ismatrix(PtPU_P,k,N))

        x = self.x
        y = self.y
        w = self.w
        assert(isvector(x,N))
        assert(isvector(y,N))
        assert(isvector(w,k))

        # Step 3: form right-hand sides
        g = sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        p = x - y + q - Phi.dot(w)
        assert(isvector(g,N))
        assert(isvector(p,N))
        
        # Step 4: Form the reduced system G dw = h
        inv_XY = 1.0/(x + y)
        A = PtPU_P * inv_XY
        assert(ismatrix(A,k,N))    
        G = (A * y).dot(Phi) - PtPUP
        assert(issquare(G,k))
        h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
        assert(isvector(h,k))
            
        # Step 5: Solve for del_w
        del_w = self.linsolve(G,h) # Uses either dense or sparse least-squares
        assert(isvector(del_w))
        assert(isvector(del_w,k))
            
            # Step 6
        Phidw= Phi.dot(del_w)
        assert(isvector(Phidw,N))
        del_y = inv_XY*(g + y*p - y*Phidw)
        assert(isvector(del_y,N))
        
        # Step 7
        del_x = del_y+Phidw-p
        assert(isvector(del_x,N))      
            
            # Step 8 Step length
        steplen = max(np.amax(-del_x/x),np.amax(-del_y/y))
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
              
        self.x = x + steplen * del_x
        self.y = y + steplen * del_y
        self.w = w + steplen * del_w    
        
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
        
        
