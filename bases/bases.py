import math
import numpy as np
import scipy
import scipy.sparse as sps
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
    def generate_basis(self,**kwargs):
        raise NotImplementedError()

class BasisModifier(object):
    def modify_basis(self,**kwargs):
        raise NotImplementedError()

class BasisWrapper(BasisGenerator):
    """
    This is an object that wraps around a state-space basis generator 
    and deals with special nodes like out of bounds nodes.

    The assumption is that a special node requires a distinct 
    Elementary vector associated with it because it is 
    sufficiently different
    """
    def __init__(self,gen_objs,modifiers=[]):
        """
        The generator function is a the object that does the actual
        generation of the basis; this object just does some 
        bookkeeping
        """
        self.generators = gen_objs
        self.modifiers = modifiers

    def add_special_modifiers(self,discretizer):
        """
        Adds a special case modifier for 
        """
        node_ids = set()
        for mapper in discretizer.exception_node_mappers:
            # Get the unique node id
            assert(1 == mapper.get_num_nodes())
            nid = mapper.get_node_ids()[0]
            assert(nid not in node_ids)
            node_ids.add(nid)

            # Build the modifier;  add to list.
            mod = SpecialCaseModifier(nid)
            self.modifiers.append(mod)

    def generate_basis(self, **kwargs):
        """
        Generates basis columns based on the provided points.
        
        Note that any special points (e.g. non-physical points) 
        will be given an elementary column vector. 
        """
        Bases = []
        for gen in self.generators:
            Bases.append(gen.generate_basis(**kwargs))
        B = np.hstack(Bases)
        (M,K) = B.shape
        assert(K <= M) # Shouldn't be a frame.

        for mod in self.modifiers:
            B = mod.modify_basis(B)

        assert(not np.any(np.isnan(B)))

        return B

class SpecialCaseModifier(BasisModifier):
    def __init__(self,node_id):
        self.node_id = node_id
        
    def modify_basis(self,B):
        (N,K) = B.shape
        nid = self.node_id

        # Row is nan
        if isinstance(B,np.ndarray):
            B[nid,:] = 0
            B = np.hstack([B,np.zeros((N,1))])
            B[nid,K] = 1
        else:
            assert(isinstance(B,sps.spmatrix))
            B_coo = B.tocoo()

            # Filter out the node_id
            mask = (B_coo.row != nid)

            # Build the new ijv format
            data = np.append(B.data[mask],1.0)
            row = np.append(B.row[mask],nid)
            col = np.append(B.col[mask],K)

            # Build dat matrix
            B = sps.coo_matrix((data,(row,col)),
                               shape=(N,K+1))
        return B
    
class ConstBasis(BasisGenerator):
    def __init__(self):
        pass
    def generate_basis(self,**kwargs):
        points = kwargs['points']
        (N,d) = points.shape
        return np.ones((N,1)) / np.sqrt(N)

class IdentityBasis(BasisGenerator):
    def __init__(self):
        pass
    def generate_basis(self,**kwargs):
        points = kwargs['points']
        (N,d) = points.shape
        return sps.eye(N)

class FromSolFileBasis(BasisGenerator):
    """
    Parrot back part of an old solution as a basis
    """
    def __init__(self,filename,block,num_states):
        n = num_states
        data = np.load(filename)
        (N,) = data.shape
        assert(0 == N % n)

        Aplus1 = N / n
        sol = np.reshape(data,(n,Aplus1),order='F')
        self.basis = sol[:,block]        
        
    def generate_basis(self,**kwargs):
        return self.basis[:,np.newaxis]    
        
        
def normalize_cols(M):
    (n,m) = M.shape
    scale = np.linalg.norm(M,axis=0)
    assert(not np.any(scale == 0))
    assert(not np.any(np.isnan(scale)))
    
    M = M / scale
    assert((n,m) == M.shape)
    assert(not np.any(np.isnan(M)))
    return M
    

    
