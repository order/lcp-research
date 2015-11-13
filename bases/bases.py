import math
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import lcp.util
import itertools
from collections import defaultdict

import time

class BasisGenerator(object):
    """
    This is an object that wraps around a "basic" basis generator and
    deals with special nodes like out of bounds nodes.

    The assumption is that a special node requires a distinct elementary
    vector associated with it because it is sufficiently different
    """
    def __init__(self,gen_obj):
        self.generator_function = gen_obj
        self.remappers = None
        """
        TODO: implement "remapper" logic so that special nodes can be mapped
        to real physical locations that behave like we'd expect the special 
        node to be similar to
        """        

    def generate(self, points,**kwargs):
        """
        Generates basis columns based on the provided points.
        
        Note that any special points (e.g. non-physical points) will 
        be given an elementary column vector. 
        """

        (N,D) = points.shape

        # Special points are things like OOB nodes    
        special_points = np.array(sorted(kwargs.get('special_points',[])))
        sp_set = set(special_points)
        S = len(sp_set)         

        # Make sure any NaN row that wasn't remapped is a special state.
        nan_mask = np.any(np.isnan(points),axis=1)
        assert((N,) == nan_mask.shape)
        nan_set = set(nan_mask.nonzero()[0])            
        assert(nan_set <= sp_set)
        # All non-phyiscal states should be special
           
        # Generate all the bases, reserving bases for the special 
        B = self.generator_function\
                .generate(points,special_points=special_points)
                
        (M,K) = B.shape
        assert(N == M)

        # Give any special points their own column
        if len(special_points) > 0:
            B[special_points,:] = 0 # Blot all special points out
            B = np.hstack([B,np.zeros((N,S))])        
            B[special_points,K:] = np.eye(S)

        return B

    def generate_block_diag(self,points,R,**kwargs):
        """
        Builds a block diagonal matrix where each block is one of R identical copies of
        a K-column basis matrix defined by the (N,d) point array, and the generator function.
        """
        (N,d) = points.shape
        B = self.generate(points,**kwargs)
        (M,K) = B.shape
        assert(N == M)
        
        RepB = [B]*R
        BDB = scipy.linalg.block_diag(*RepB)
        assert((N*R,K*R) == BDB.shape)
        return BDB

class BasicBasisGenerator(object):
    """
    The BasisGenerator uses these for actually doing to mapping of rows to basis values.
    It handles some of the higher-level concerns like cleaning and special point handling
    """
    def __init__(self):
        pass
    def generate(self,points,**kwargs):
        raise NotImplementedError()

class RandomFourierBasis(BasicBasisGenerator):
    def __init__(self,**kwargs):
        self.scale = kwargs.get('scale',1.0)
        self.K = kwargs['num_basis']
        assert(self.K >=1)
    def generate(self,points,**kwargs):
        K = self.K
        (N,D) = points.shape
        special_points = kwargs.get('special_points',[])

        # Format: B = [1 | fourier columns]
        W = self.scale * np.random.randn(D,K-1) # Weights
        Phi = 2.0 * np.pi * np.random.rand(K-1) # Phases shift
        F = np.sin(points.dot(W) + Phi) # Non-constant columns
        assert((N,K-1) == F.shape)
        B = np.hstack([np.ones((N,1)),F])
        assert((N,K) == B.shape)

        # Mask out any special points
        if len(special_points) > 0:
            B[special_points,:] = 0
        
        # Normalize        
        normalize_cols(B)
        
        return B
        
        
class RegularRadialBasis(BasicBasisGenerator):
    def __init__(self,**kwargs):
        self.centers = kwargs['centers']
        self.bw = kwargs['bandwidth']
    def generate(self,points,**kwargs):        
        (N,D) = points.shape

        special_points = kwargs.get('special_points',[])
        B = []
        (K,d) = self.centers.shape
        assert(d == D)
        for i in xrange(K):
            mu = self.centers[i,:]
            assert((D,) == mu.shape)
            B.append(gaussian_rbf(points,mu,self.bw))
        
        B = np.column_stack(B)
        assert((N,K) == B.shape)
        B[special_points,:] = 0
        
        # Normalize        
        normalize_cols(B)
        
        return B
        
def normalize_cols(M):
    (n,m) = M.shape
    scale = np.linalg.norm(M,axis=0)
    assert(not np.any(scale == 0))
    assert(not np.any(np.isnan(scale)))
    
    M = M / scale
    assert((n,m) == M.shape)
    assert(not np.any(np.isnan(M)))
    return M
    
def gaussian_rbf(X,mu,bw):
    C = X - mu
    assert(C.shape == X.shape)
    RBF = np.exp(-np.power(np.linalg.norm(C,axis=1) / bw, 2))
    assert(RBF.shape[0] == X.shape[0])
    return RBF
    