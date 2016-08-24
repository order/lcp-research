import numpy as np
import discretize
import itertools

from utils import row_vect,col_vect,is_int

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
        
        self.physical_max_index = np.product(lens) - 1
        self.max_index = self.physical_max_index + 2*D
        self.lens = np.array(lens)
          
        self.coef = c_stride_coef(lens)

    def get_num_nodes(self):
        return self.max_index + 1

    def get_num_physical_nodes(self):
        return self.physical_max_index + 1

    def get_oob_indices(self):
        """
        Get all the indices for OOB nodes
        """
        ret =  range(self.physical_max_index+1,self.max_index+1)
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
        return int(self.physical_max_index + 2*(d+1) + (sign - 1)/2)
    
    def coords_to_indices(self,coords):
        """
        Turn D dimensional coords into indices
        """
        (N,D) = coords.shape
        assert D == self.dim # Dimension right
        assert not np.any(np.isnan(coords))

        # Does most of the work
        
        indices = coords.dot(self.coef)

        # OOB handling
        for d in xrange(D-1,-1,-1):
            # Reverse order so higher violations are masked by lower ones.
            # Too small
            oob_idx = self.get_oob_index(d,-1)
            mask = coords[:,d] < 0
            indices[mask] = oob_idx

            # Too large
            oob_idx = self.get_oob_index(d,1)
            mask = (coords[:,d] >= self.lens[d])
            indices[mask] = oob_idx
            
        return indices

    def indices_to_coords(self,indices):
        # Converts indices to coordinates
        assert is_int(indices)
        (N,) = indices.shape
        D = len(self.coef)

        # Does the hard work
        coords = np.empty((N,D))
        res = indices
        for d in xrange(D):
            (coord,res) = divmod(res,self.coef[d])
            coords[:,d] = coord

        # OOB indices mapped to NAN
        oob_mask = np.logical_or(indices < 0,
                                 indices > self.physical_max_index)
        coords[oob_mask] = np.nan        
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
        return indices > self.physical_max_index

##################
# MISC FUNCTIONS #
##################

# Coord to index testing function (slow)
def slow_coord_to_index(target,lens):
    """
    Slow but simple way of converting coords to indices
    C-style indexing
    """
    assert 1 == target.ndim
    assert 1 == lens.ndim
    D = lens.size
    assert D == target.size
    
    coord = np.zeros(D)
    I = 0
    while np.any(target > coord):
        coord[-1] += 1
        I += 1
        for d in xrange(D-1,-1,-1):
            if lens[d] == coord[d]:
                coord[d] = 0
                if d > 0:
                    coord[d-1] += 1
                else:
                    return -1
            else:
                break        
    return I

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
