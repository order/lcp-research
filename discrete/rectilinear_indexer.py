import numpy as np
import discretize
import itertools

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
            mask = (coords[:,d] >= self.lens[d]-1)
            indices[mask] = oob_idx
            
        return indices

    def indices_to_coords(self,indices):
        # Converts indices to coordinates
        assert discretize.is_int(indices)
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
                                 indices >= self.physical_max_index)
        coords[oob_mask] = np.nan
        
        return coords.astype(np.int64)

    def cell_shift(self):
        # Returns the index offsets requiresd to visit all nodes in a cell
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
        
        neighbors = indices[:,np.newaxis] + shift[np.newaxis,:]
        assert((N,2**D) == neighbors.shape)
        return neighbors


    def is_oob(self,indices):
        return indices > self.physical_max_index
