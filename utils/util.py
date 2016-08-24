import numpy as np
import time
import os
import copy
import math
import operator
import inspect

import scipy.sparse
import matplotlib.pyplot as plt

import importlib

def is_sorted(x):
    """
    Check if an array is sorted
    """
    if isinstance(x,np.ndarray):
        (N,) = x.shape
    else:
        N = len(x)
    return all([x[i] <= x[i+1] for i in xrange(N-1)])

def is_vect(x):
    if not isinstance(x,np.ndarray):
        return False
    return 1 == x.ndim

def is_mat(x):
    if not isinstance(x,np.ndarray):
        return False
    return 2 == x.ndim

def is_int(x):
    """
    Checks if a ndarray is filled with integers
    """
    assert isinstance(x,np.ndarray)
    if issubclass(x.dtype.type,np.integer):
        return True
    f = np.mod(x,1.0)
    mask = ~np.isnan(x) # ignore NaN
    return np.all(f[mask] < 1e-15)

def make_points(gens,ret_mesh=False,order='C'):
    """
    Makes the mesh in the order you would expect for
    np.reshape after.

    E.g. if handed [np.linspace(0,1,5),np.linspace(0,1,7)]
    then this would make the points P such that if mapped
    the 2D plot makes spacial sense. So np.reshape(np.sum(P),(5,7))
    would look pretty and not jumpy
    """
    if 'F' == order:
        gens = list(reversed(gens))
    if 1 == len(gens):
        return gens[0][:,np.newaxis] # meshgrid needs 2 args
    
    meshes = np.meshgrid(*gens,indexing='ij')
    points = np.column_stack([M.flatten() for M in meshes])
    if 'F' == order:
        return np.fliplr(points)
    if ret_mesh:
        return points,meshes
    return points

def banner(msg):
    """
    Take a look at banner, Michael!

    Loudly announces something, and provide limited introspection
    about what requested the banner.
    """
    
    # Introspect
    f = inspect.currentframe().f_back
    mod_str = f.f_code.co_filename
    mods = mod_str.split(os.sep)
    mod = os.sep.join(['...'] + mods[-2:]) # Use last 2 path elems
    lineno = f.f_lineno
    loc_msg = 'From: {0}:{1}'.format(mod,lineno)

    # Pad
    N = max(len(loc_msg), len(msg))
    msg += ' '*(N - len(msg))
    loc_msg += ' '*(N - len(loc_msg))
    assert(len(msg) == len(loc_msg))
    
    # Print
    print '#'*(N+4)
    print '# ' + msg + ' #'
    print '# ' + loc_msg + ' #'
    print '#'*(N+4)      

def col_vect(v):
    """
    Convert a n-vector into a nx1 np array.
    """
    assert(len(v.shape) == 1)
    return v[:,np.newaxis]

def row_vect(v):
    """
    Convert a n-vector into a 1xn np array.
    """
    assert(len(v.shape) == 1)
    return v[np.newaxis,:]    
