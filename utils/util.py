import numpy as np
import time
import os
import copy
import math
import operator
import inspect

import scipy.sparse
import matplotlib.pyplot as plt

import h5py
import importlib

def top_k_value(q,k,thresh):
    assert(q.size > k > 0)
    sq = np.sort(q.flatten())
    N = len(sq)
    assert(N == q.size)
    
    for i in xrange(N-k-1,N):
        if sq[i] >= thresh:
            return sq[i]
    return sq[-1]

def tail_max(discount,T):
    return np.power(discount,T) / (1.0 - discount)

def bounded_tail(discount,bound):
    T = np.ceil(np.log((1.0 - discount)*bound) / np.log(discount))
    assert(bound > tail_max(discount,T))
    return int(T)

def save_ndarray_hdf5(filename,A):
    f = h5py.File(filename,'w')
    dset = f.create_dataset("dataset", data=A)
    f.close()

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

def sparsity_ratio(A):
    return float(A.nnz) / float(np.prod(A.shape))

def shift_array(x,pos,pad=np.nan):        
    (N,) = x.shape # Just for array atm
    
    shift = np.empty(N)
    if pos >= 0:
        # Right shift:
        # [0 0 0 x x ... x]
        shift[:pos] = pad
        shift[pos:] = x[:-pos]
    else:
        # Left shift:
        #[x ... x x 0 0 0]
        shift[:pos] = x[-pos:]
        shift[pos:] = pad
    
    return shift

def hash_ndarray(A):
    return hash(A.tostring())
    
def split_extension(filename,ext=None):
    if ext:
        assert(not ext.startswith('.'))
        assert(filename.endswith('.' + ext))

    comps = filename.split('.')
    return ('.'.join(comps[:-1]),comps[-1])
    

def kwargify(**kwargs):
    # I like this dict format
    return kwargs

def get_instance_from_file(conf_file):
    """
    Loads a class from file string
    So if the string is 'foo/bar/baz.py' then it loads the UNIQUE
    class in that file.
    """
    module = load_module_from_filename(conf_file)
    classlist = list_module_classes(module)

    assert(1 == len(classlist)) # Class is UNIQUE.
    return classlist[0][1]() # Instantiate too

def load_module_from_filename(filename):
    """
    Loads a module from a relative filename
    e.g. config/solvers/foo.py will load
    config.solvers.foo
    
    """
    assert(filename.endswith('.py'))
    module_str = filename[:-3].replace(os.sep,'.')
    module = importlib.import_module(module_str)
    return module

def list_module_classes(mod):
    classes = inspect.getmembers(mod, inspect.isclass)
    return classes

# Some debugging routines
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

