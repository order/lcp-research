import math
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import utils.parsers

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import lcp.util
import itertools
from collections import defaultdict

import time

class BasisGenerator(object):
    """
    This is an object that wraps around a "basic" basis generator 
    and deals with special nodes like out of bounds nodes.

    The assumption is that a special node requires a distinct 
    Elementary vector associated with it because it is 
    sufficiently different
    """
    def __init__(self,gen_obj):
        """
        The generator function is a the object that does the actual
        generation of the basis; this object just does some 
        bookkeeping
        """
        self.basic_generator = gen_obj


    def generate_basis(self, points,**kwargs):
        """
        Generates basis columns based on the provided points.
        
        Note that any special points (e.g. non-physical points) 
        will be given an elementary column vector. 
        """

        parser = utils.parsers.KwargParser()
        parser.add('special_points',[])
        args = parser.parse(kwargs)

        (N,D) = points.shape

        # Special points are things like OOB nodes
        unsorted_points = args.get('special_points',[])
        special_points = np.array(sorted(unsorted_points))
        sp_set = set(special_points)
        S = len(sp_set)         

        # Make sure any NaN row that wasn't remapped is special
        nan_mask = np.any(np.isnan(points),axis=1)
        assert((N,) == nan_mask.shape)
        nan_set = set(nan_mask.nonzero()[0])
        assert(nan_set <= sp_set)
        # All non-phyiscal states should be special
           
        # Generate all the bases, reserving bases for the special 
        B = self.basic_generator\
                .generate_basis(points,special_points=special_points)
                
        (M,K) = B.shape
        assert(N == M)

        # Give any special points their own column
        if len(special_points) > 0:
            B[special_points,:] = 0 # Blot all special points out
            B = np.hstack([B,np.zeros((N,S))])        
            B[special_points,K:] = np.eye(S)
            
        #B = normalize_cols(B)
        return B

    def generate_block_diag(self,points,R,**kwargs):
        """
        Builds a block diagonal matrix where each block is one of 
        R identical copies of a K-column basis matrix defined by 
        the (N,d) point array, and the generator function.
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
    The BasisGenerator uses these for actually doing to mapping of 
    rows to basis values. It handles some of the higher-level 
    concerns like cleaning and special point handling
    """
    def __init__(self):
        pass
    def isortho(self):
        raise NotImplementedError()
    def generate_basis(self,points,**kwargs):
        raise NotImplementedError()

class IdentityBasis(BasicBasisGenerator):
    def __init__(self):
        pass
    def isortho(self):
        return True
    def generate_basis(self,points,**kwargs):
        # Parse kwargs
        parser = utils.parsers.KwargParser()
        parser.add('special_points',[])
        args = parser.parse(kwargs)
        special_points = args['special_points'] 
        (N,D) = points.shape

        num_normal = N - len(special_points)
        normal_mask = np.ones(N,dtype=bool)
        normal_mask[special_points] = False
        
        B = np.empty((N,num_normal))
        B[normal_mask,:] = np.eye(num_normal)
        B[~normal_mask,:] = 0
                
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
    

    
