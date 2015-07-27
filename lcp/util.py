import numpy as np
import time
import os
import copy
import math

import scipy.sparse
import matplotlib.pyplot as plt

# Utilities; put in another file?

def debug_mapprint(level,**kwargs):
    if level:
        for (k,v) in kwargs.items():
            print k,'=',v

def debug_print(level,str):
    if level:
        print str

def shape_str(M):
    return 'x'.join(map(str,M.shape))

def col_vect(v):
    """
    Convert a n-vector into a nx1 np array.
    This is an annoying distinct in numpy.
    """
    assert(len(v.shape) == 1)
    return v[:,np.newaxis]    

def max_eigen(M):
    """
    Find the max eigenvalue; wraps around a simple scipy call
    """
    return scipy.sparse.linalg.eigs(M,k=1,return_eigenvectors=False)

def quad(a,b,c):
    """
    Solve a simple 1d quadratic formula
    """
    d = b**2 - 4*a*c
    if d < 0:
        # Only report real solutions
        return None
    return ((-b + math.sqrt(d))/2,(-b - math.sqrt(d))/2)

def has_pos_diag(M):
    """
    Check if diagonal is positive
    """
    [n,m] = M.shape
    assert(n==m)
    for i in xrange(n):
        if M[i,i] <= 0:
            return False
    return True

def isvector(x):
    S = x.shape
    return len(S) == 1
    
def issquare(x):
    S = x.shape
    
    return len(S) == 2 and (S[0] == S[1])
    
def nonneg_proj(x):
    """
    Projections onto non-negative orthant; the []_+ operator
    """
    assert(isvector(x))
    return np.maximum(np.zeros(x.size),x)
    

def proj_forward_prop(x,M,q,w):
    """
    Does the projected version of Gauss-Siedel-ish forward solving
    """
    N = len(x)
    y = np.zeros(N)
    for i in xrange(N):
        curr_round_terms = M[i,0:i].dot(y[0:i])
        past_round_terms = M[i,i:].dot(x[i:])
        y[i] = max(0,x[i] - w * (1/M[i][i]) * (q[i] + curr_round_terms  + past_round_terms))
    return y
    
def basic_residual(x,w):
    """
    A basic residual for LCPs; will be zero if 
    """
    res = np.minimum(x,w)
    return np.linalg.norm(res)
    
# Fisher-Burmeister residual
def fb_residual(x,w):
    fb = np.sqrt(x**2 + w**2) - x - w
    return np.linalg.norm(fb)
    
###############################
# State
    
class State(object):
    """An object representing a point in the state-space of an LCP
    Note that the point may not be feasible or complementary
    (and so w != Mx+q)
    We have a separate field Mxq to cache Mx+q
    """
    def __init__(self,**kwargs):
        self.x = None # 
        self.w = None
        self.Mxq = None
        self.iter = 0
    def set_x(self,x):
        self.x = x
        self.Mxq = None
    def cache_Mxq(self,mxq):
        self.Mxq = mxq
    def inc(self):
        self.iter+=1
        
###############
# Record stuff
class Record(object):
    def __init__(self):
        pass
    

def null_recorder(record,state):
    pass
    
def residual_recorder(record,state):
    if not hasattr(record,'residual'):
        record.residual = []
    record.residual.append(basic_residual(state.x,state.w))

def dot_records(record,state):
    if not hasattr(record,'dots'):
        record.dots = []
    record.dots.append(state.x.dot(state.w))       
  
def state_recorder(record,state):
    if not hasattr(record,'states'):
        record.states = []
    record.states.append(np.array(state.x))

def support_recorder(record,state):
    if not hasattr(record,'support'):
        record.support = []
    support = sum(state.x > 0)
    record.support.append(support)
    
    
##########################
# Termination functions
    
def time_term(time_lim,state):
    if os.name == 'nt':
        t = time.clock()
    else:
        t = time.time()

    if not hasattr(state,'start_time'):
        state.start_time = t
        return False
    return (t - state.start_time) >= time_lim
    
def max_iter_term(max_iter,state):
    return state.iter >= max_iter
    
def res_thresh_term(res_fn,thresh,state):
    if state.iter == 0:
        return False
    return res_fn(state.x,state.w) <= thresh
    
##############################
# Params 

class Params(object):
    pass
    
    
#############################
# Plotting for records

def plot_state_img(record,**kwargs):
    """ Plots the state trajectory as an image
    """
    max_len = kwargs.get('max_len',2**64-1)    
    S = np.array(record.states)
    f,ax = plt.subplots()
    Img = (S - record.states[-1]) / (record.states[-1] + 1e-12) + 1e-12
    l = min(max_len,Img.shape[0])
    ax.imshow(np.log(np.abs(Img[:l,:].T)), interpolation='nearest')
    ax.set_title('Relative log change from final iter')
    plt.show()
    
