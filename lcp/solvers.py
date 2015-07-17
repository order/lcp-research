import numpy as np
import time
import os
import copy
import math

import scipy.sparse
import matplotlib.pyplot as plt
import math
from lcp.util import *
import mdp.mdp as mdp

##############################
# MDP block split
def mdp_value_iter(M,q,state,**kwargs):  
    MDP = kwargs['MDP']
    
    N = q.size
    n = MDP.num_states
    A = MDP.num_actions
    x = np.ones(N)
    y = np.ones(N)
    
    gamma = MDP.discount
    v = np.ones((n,1))
    
    P_T = []
    for a in xrange(MDP.num_actions):
        # Transpose, and convert to csr sparse
        P_T.append(scipy.sparse.csr_matrix(MDP.transitions[a].T))
        
    I = 0
    while True:
        # Kind of ver
        Vs = []
        for a in xrange(MDP.num_actions):            
            v_a = col_vect(MDP.costs[a]) + gamma*P_T[a].dot(v)
            #print 'v_a',v_a.shape
            Vs.append(v_a.flatten())
        Vs = np.array(Vs).T
        #print 'Vs',Vs.shape        
        #print np.amin(Vs,axis=1).shape
        v = col_vect(np.amin(Vs,axis=1))
        
        # Construct the x vector from v
        x[:n] = v.flatten()
        assert(v.size == n) 
        mask = (Vs == v)
        ties = np.sum(mask,axis=1) # tie break uniformly?
        x[n:] = ((mask.T) / ties).flatten() # Dual variables
        
        state.x = x
        state.w = M.dot(x) + q
        state.iter = I
        I += 1
        yield state
        
    

##############################
# MDP block split
def mdp_ip_iter(M,q,state,**kwargs):
    sigma = kwargs.get('centering_coeff',0.1)
    beta = kwargs.get('linesearch_backoff',0.8)
    
    N = q.size
    x = np.ones(N)
    y = np.ones(N)
    
    MDP = kwargs['MDP']
    split = mdp.MDPValueIterSplitter(MDP)

    I = 0
    OI = 0
    while True:
        #print 'Outer loop iter',OI
        #print '* Outer Residual:\t', np.linalg.norm(y - (q + M.dot(x)))
        #print '* Complementarity:\t',x.dot(y)

        # Outer loop solved by solving different LCP(B,q_k)
        (B,q_k) = split.update(x)
        state = State()

        inner_solver = kojima_ip_iter(B,q_k,state,**kwargs)   
        II = 0        
        while x.dot(y) >= 1e-6 or np.linalg.norm(y - (q_k + B.dot(x))) > 1e-6:
            #print '\tInner loop iter',II
            #print '\t* Inner Residual:', np.linalg.norm(y - (q_k + B.dot(x)))
            #print '\t* InnerComplementarity:',x.dot(y)

            state = inner_solver.next()
            x = state.x
            y = state.w
            II += 1
        state.w = M.dot(x) + q
        state.iter = OI
        yield state  
        OI += 1
        

##############################
# Kojima LCP

def kojima_ip_iter(M,q,state,**kwargs):
    sigma = kwargs.get('centering_coeff',0.1)
    beta = kwargs.get('linesearch_backoff',0.8)   
    
    N = q.size
    x = np.ones(N)
    y = np.ones(N)
    
    dot = float('inf')
    I = 0
    Bottom = scipy.sparse.hstack([-scipy.sparse.csr_matrix(M),scipy.sparse.eye(N)])
    X = scipy.sparse.diags(x,0)
    Y = scipy.sparse.diags(y,0)
    Top = scipy.sparse.hstack((Y, X))
    A = scipy.sparse.vstack([Top,Bottom]).tocsr()

    while True:
        assert(not np.any(x < 0))
        assert(not np.any(y < 0))
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)
        
        # Set up Newton direction equations         
        b = np.concatenate([sigma * dot / float(N) * np.ones(N) - x*y, r])
        X = scipy.sparse.diags(x,0)
        Y = scipy.sparse.diags(y,0)
        Top = scipy.sparse.hstack((Y, X))
        A = scipy.sparse.vstack([Top,Bottom]).tocsr()
            
        # Solve Newton direction equations
        #start = time.time()
        assert(scipy.sparse.issparse(A))
        dir = scipy.sparse.linalg.spsolve(A,b)
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
def basic_ip_iter(M,q,state,**kwargs):
    N = q.size
    k = 0
    Top = scipy.sparse.hstack([M,-scipy.sparse.eye(M.shape[0])])
    x = np.ones(N)
    s = np.ones(N)
    sigma = kwargs.get('centering_coeff',0.01)
    beta = kwargs.get('linesearch_backoff',0.9)
    
    while True:
        assert(not any(x < 0))
        assert(not any(s < 0))
        
        mu = x.dot(s) / N # Duality measure
        
        # Construct matrix from invariant Top and diagonal bottom
        X = scipy.sparse.diags(x,0)
        S = scipy.sparse.diags(s,0)
        Bottom = scipy.sparse.hstack([S, X])
        A = scipy.sparse.vstack([Top,Bottom]).tocsr()
        r = s - M.dot(x) - q
        centering = -x*s + sigma*mu*np.ones(N)

        b = np.hstack([r, centering])
        
        # Solve, break solution into two parts
        d = scipy.sparse.linalg.spsolve(A,b)        
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
    
def euler_iter(M,q,state,**kwargs):
    N = q.size
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
        
        
def euler_barrier_iter(M,q,state,**kwargs):
    N = q.size
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
        
def euler_linesearch_iter(M,q,state,**kwargs):

    wolfe_const = kwargs.get('wolfe_const',1e-4)
    step_decay = kwargs.get('step_decay',0.9)
    min_step = kwargs.get('min_step', 1e-4)
    N = q.size
    
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
        
def euler_speedy(M,q,state,**kwargs):
    N = q.size
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
    
def projected_jacobi_iter(M,q,state,**kwargs):
    assert(has_pos_diag(M))
    N = q.size
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
        
def psor_iter(M,q,state,**kwargs):
    assert(has_pos_diag(M))
    N = q.size
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
        
def extragrad_iter(M,q,state,**kwargs):
    N = q.size
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
def accelerated_prox_iter(M,q,state,**kwargs):
    N = q.size
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
    
        
###################
# Generic iteration solver   

class iter_solver():
    def __init__(self):
        self.step = 1e-3;
        self.record_fn = None
        self.term_fn = None
        self.iter_fn = None
        self.params = {}
        
    def solve(self,M,q,**kwargs):
        assert(check_lcp(M,q))        
        N = q.size
        
        # Init
        record = Record()
        state = State()
        state.x = np.zeros(N)
        state.w = np.zeros(N)
            
        args = self.params.copy()
        args.update(kwargs)
        iter = self.iter_fn(M,q,state,**args)
        while True:
            for tf in self.term_fns:
                if tf(state):
                    print 'Termination reason:',tf.func.__name__
                    return (record,state)
            state = iter.next()
            for rf in self.record_fns:
                rf(record,state)
        
        
        
