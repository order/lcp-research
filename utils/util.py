import numpy as np
import time
import os
import copy
import math
import operator

import scipy.sparse
import matplotlib.pyplot as plt

import importlib

def load_class(mod_str):
    if '.' not in mod_str:
        return eval(mod_str)

    splits = mod_str.split('.')
    mod = importlib.import_module('.'.join(splits[:-1]))
    return eval('mod.{0}'.format(splits[-1]))
    

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
    
