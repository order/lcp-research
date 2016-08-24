import numpy as np
import scipy.sparse as sps

from rectilinear_indexer import Indexer

from utils import is_vect,is_mat,row_vect,col_vect   

############
# OOB DATA #
############

class OutOfBounds(object):
    def __init__(self):
        self.data = None
        self.rows = None
        self.mask = None
        self.indices = None

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
        self.rows = np.where(oob_mask) # Components that aren't nan
        self.indices = indices[oob_mask] # Cache of the indices

        # Go through the non nan indices, unpack into data
        data = sps.lil_matrix((N,D),dtype=np.integer)
        for d in xrange(D):
            # Even indices
            mask = (self.indices == 2*d)
            rows = self.row[mask]
            self.data[mask,d] = -1

            # Odd indices
            mask = (self.indices == 2*d+1)
            rows = self.row[mask]
            self.data[mask,d] = 1

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
        
        # Points that violate some boundary
        self.rows = (self.data.tocoo()).row

        # Mask of same
        self.mask = np.zeros(N,dtype=bool)
        self.mask[self.rows] = True

        # Sanity check
        assert np.all(self.mask == grid.are_points_oob(points))

        # Pre-offset oob node or cell indices
        self.indices = self.find_oob_index()
        
    def has_oob(self):
        if self.data is None:
            # Dummy object
            return False
        
        # Are there any oob points?
        ret = self.rows.size > 0
        assert ret == (self.data.sum() > 0)
        return ret
        
    def find_oob_index(self):
        """
        Linearizes the oob information
        """
        (N,) = self.rows.shape
        indices = np.empty(N)
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
