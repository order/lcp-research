import numpy as np
import discretize
import itertools

from utils import row_vect,col_vect,is_int,is_vect

class Indexer(object):
    """
    Indexer for rectilinear grids.
    Class that maps ND integer coordinates to 1D index,
    And does the inverse.

    Assumes some function has already taken the ND points and converted
    them into grid coordinates.

    NB: this is a pretty simple operation made a little
    more complicated by out-of-bound (OOB) node handling.

    """
    def __init__(self,lens):
        lens = np.array(lens) 
        (D,) = lens.shape
        self.dim = D

        """
        There are 2 oob nodes for every dimension: too small, and too big.
        If a point is oob in several dimensions, it's assigned to the oob 
        node in the first dimension that it violates
        """
        
        self.spatial_max_index = np.product(lens) - 1
        self.max_index = self.spatial_max_index + 2*D
        self.lens = np.array(lens)
          
        self.coef = c_stride_coef(lens)

    def get_num_nodes(self):
        return self.max_index + 1

    def get_num_spatial_nodes(self):
        return self.spatial_max_index + 1

    def get_oob_indices(self):
        """
        Get all the indices for OOB nodes
        """
        ret =  range(self.spatial_max_index+1,self.max_index+1)
        assert 2*self.dim == len(ret)
        return ret

    def get_oob_index(self,d,sign):
        """
        Get a particular OOB index for a dimension and sense
        Too small = -1
        Too big = +1
        """
        assert sign in [-1,1]
        assert 0 <= d < self.dim
        return int(self.spatial_max_index + 2*(d+1) + (sign - 1)/2)
    
    def coords_to_indices(self,coords):
        """
        Turn D dimensional coords into indices
        """
        (N,D) = coords.shape
        assert D == self.dim # Dimension right

        # Does most of the work

        raw_coords = coords.coords
        oob = coords.oob

        if oob.has_oob():
            indices = np.empty(N)
            indices[~oob.mask] = (raw_coords[~oob_mask,:]).dot(self.coef)
            oob_offset = self.get_num_spatial_nodes() 
            indices[oob.mask] = oob.indices + oob_offset
        else:
            indices = raw_coords.dot(self.coef)
        
        return indices

    def indices_to_coords(self,indices):
        # Converts indices to coordinates
        assert is_vect(indices)
        assert is_int(indices)
        
        (N,) = indices.shape
        D = len(self.coef)

        # Does the hard work
        raw_coords = np.empty((N,D))
        res = indices
        for d in xrange(D):
            (coord,res) = divmod(res,self.coef[d])
            raw_coords[:,d] = coord

        # OOB indices mapped to NAN
        oob_mask = self.are_indices_oob(indices)
        raw_coords[oob_mask,:] = np.nan

        oob_indices = indices - self.get_num_spatial_nodes()
        oob_indices[~oob_mask] = np.nan

        oob = OutOfBounds()
        oob.build_from_oob_indices(oob_indices,D)

        Coordinates(raw_coords,oob)
        
        return coords

    def cell_shift(self):
        # Returns the index offsets required to visit all nodes in a cell
        # o - o
        # |   |
        # x - o
        D = self.dim
        shift = np.empty(2**D)

        for (i,diff) in enumerate(itertools.product([0,1],repeat=D)):
            diff = np.array(diff)
            shift[i] = diff.dot(self.coef)
            
        return shift

    def get_neighbors(self,indices):
        (N,) = indices.shape
        D = self.dim
        shift = self.cell_shift()
        
        neighbors = col_vect(indices) + row_vect(shift)
        assert((N,2**D) == neighbors.shape)
        return neighbors
    
    def are_coords_oob(self,coords):
        (N,D) = coords.shape
        assert D == self.dim
        L = np.any(coords < 0,axis=1)
        U = np.any(coords >= row_vect(self.lens),axis=1)
        return np.logical_or(L,U)
            
    def are_indices_oob(self,indices):
        assert 1 == indices.ndim
        assert not np.any(indices > self.max_index)
        assert not np.any(indices < 0)
        return indices > self.spatial_max_index

##################
# MISC FUNCTIONS #
##################

# Coord to index testing function (slow)
def slow_coord_to_index(target,lens):
    """
    Slow but simple way of converting coords to indices
    Uses C-style indexing; this means the last coordinate
    changes most freqently.

    For an (P,Q,R) matrix: 
    0 0 0 -> 0
    0 0 1 -> 1
    0 0 2 -> 2
       ...
    p q r -> r + R*q + (R*Q)*p
    """
    assert is_vect(target)
    assert is_vect(lens)
    assert target.shape == lens.shape
    (D,) = lens.shape

    idx = 0
    mult = 1
    for d in xrange(D-1,-1,-1):
        idx += mult * target[d]
        mult *= lens[d]
    return idx

def even_slower_coord_to_index(target,lens):
    assert is_vect(target)
    assert is_vect(lens)
    assert target.shape == lens.shape

    N = np.prod(lens)
    C = np.reshape(np.arange(N),lens) # Should be row-major ordering
    return C[tuple(target)]

def c_stride_coef(lens):
    """
    The coefficients for converting ND-array coords
    to 1D array coords

    NB: in row-major (C) order; requires some flipping.
    """
    coef = np.cumprod(np.flipud(lens))
    coef = np.roll(coef,1)
    coef[0] = 1.0
    coef = np.flipud(coef)
    return coef
