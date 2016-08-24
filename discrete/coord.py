import numpy as np
import scipy.sparse as sps

from utils import is_int,is_vect,is_mat,row_vect,col_vect   

############
# OOB DATA #
############

class OutOfBounds(object):
    def __init__(self):
        self.data = None    # sps.spmatrix
        self.mask = None    # np.array, dtype=bool
        self.indices = None # np.array

        self.dim = None
        self.num = None
        self.shape = None
    
    def build_from_oob_indices(self,indices,D):
        """
        Indices should be np.nan if not oob.
        Max spatial index should be already subtracted off, so indices should 
        be integers in [0,2*D).
        """
        
        assert is_vect(indices)
        assert is_int(indices) # ignore nan

        oob_mask = ~np.isnan(indices)
        
        assert not np.any(indices[oob_mask] < 0)
        assert not np.any(indices[oob_mask] >= 2*D)

        (N,) = indices.shape
        self.dim = D
        self.num = N
        self.shape = (N,D)

        self.mask = oob_mask # Binary mask
        self.indices = np.empty(N)
        self.indices[~oob_mask].fill(np.nan)
        self.indices[oob_mask] = indices[oob_mask] # Cache of the indices
        
        # Go through the non nan indices, unpack into data
        data = sps.lil_matrix((N,D),dtype=np.integer)
        for d in xrange(D):
            # Even indices
            mask = (self.indices == 2*d)
            data[mask,d] = -1

            # Odd indices
            mask = (self.indices == 2*d+1)
            data[mask,d] = 1
        self.data = data.tocsc()
            
    def build_from_points(self,grid,points):
        assert is_mat(points)
        (N,D) = points.shape
        
        self.dim = D
        self.num = N
        self.shape = (N,D)
        
        low = grid.get_lower_boundary()
        high = grid.get_upper_boundary()

        # What boundary is violated;
        # -1 lower boundary,
        # +1 upper boundary
        U = sps.csc_matrix(points > high,dtype=np.integer)
        L = sps.csc_matrix(points < row_vect(low),dtype=np.integer)
        self.data = U - L
        
        # Mask of same
        self.mask = np.zeros(N,dtype=bool)
        self.mask[self.data.nonzero()[0]] = True
        assert isinstance(self.mask,np.ndarray)
        assert (N,) == self.mask.shape

        # Sanity check
        assert np.all(self.mask == grid.are_points_oob(points))

        # Pre-offset oob node or cell indices
        self.indices = self.find_oob_index()
        assert is_vect(self.indices)
        assert np.all(np.isnan(self.indices) == ~self.mask)
        
    def has_oob(self):
        assert self.data is not None
        
        # Are there any oob points?
        ret = (self.data.nnz > 0)
        assert ret == (np.sum(self.mask) > 0)
        return ret
        
    def find_oob_index(self):
        """
        Linearizes the oob information
        """
        indices = np.empty(self.num)
        indices.fill(np.nan)
        # Reverse order, larger overwrites smaller
        for d in xrange(self.dim-1,-1,-1):
            # All points that are too small in this dimension
            small = (self.data.getcol(d) == -1).nonzero()[0]
            indices[small] = 2*d

            # ALl points that are too large in this dimension
            large = (self.data.getcol(d) == 1).nonzero()[0]
            indices[large] = 2*d + 1
        return indices

        

###################
# COORDINATE DATA #
###################

class Coordinates(object):
    def __init__(self,coords,oob):
        assert is_mat(coords)
        assert isinstance(oob,OutOfBounds)
        
        (N,D) = coords.shape
        assert oob.dim == D
        assert oob.num == N

        self.dim = D
        self.num = N
        self.shape = (N,D)
        
        self.coords = coords
        self.oob = oob   
