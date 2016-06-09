import numpy as np
import discretize
import itertools

class Indexer(object):
    def __init__(self,lens,order='C'):
        lens = np.array(lens)
        (D,) = lens.shape
        self.dim = D

        # There are 2 oob nodes for every dimension.
        # Too small, and too big.
        # If a point is oob in several dimensions, it's assigned to the oob node in the
        # first dimension that it violates
        self.physical_max_index = np.product(lens) - 1
        self.max_index = self.physical_max_index + 2*D
        self.lens = np.array(lens)
        
        """
        Assume C ordering most of the time.
        Fortran style just for completeness
        """
        assert(order=='C' or order=='F')
        self.order = order
        if order == 'F':
            # Fortran order; first coord changes index least
            coef = np.cumprod(lens)
            coef = np.roll(coef,1)
            coef[0] = 1.0
        else:
            # C order; last coord changes index least
            # Need to flip the coefficients around to
            # make the C order            
            coef = np.cumprod(np.flipud(lens))
            coef = np.roll(coef,1)
            coef[0] = 1.0
            coef = np.flipud(coef)            
        self.coef = coef

    def get_oob_indices(self):
        ret =  range(self.physical_max_index+1,
                     self.max_index+1)
        assert(2*self.dim == len(ret))
        return ret

    def get_oob_index(self,d,sign):
        assert(sign in [-1,1])
        assert(0 <= d < self.dim)
        
        return int(self.physical_max_index + 2*(d+1) + (sign - 1)/2)
    
    def coords_to_indices(self,coords):
        """
        Turn D dimensional coords into indices
        """
        (N,D) = coords.shape
        assert(D == self.dim) # Dimension right
        assert(not np.any(np.isnan(coords)))
        
        indices = coords.dot(self.coef)
        for d in xrange(D-1,-1,-1):
            # Reverse order so higher violations are masked by lower ones.
            # Too small
            oob_idx = self.get_oob_index(d,-1)
            mask = coords[:,d] < 0
            indices[mask] = oob_idx

            # Too large
            oob_idx = self.get_oob_index(d,1)
            mask = coords[:,d] >= self.lens[d]
            indices[mask] = oob_idx 
        return indices
    
    def indices_to_coords(self,indices):
        (N,) = indices.shape
        D = len(self.coef)
        
        assert(discretize.is_int(indices))     
        res = indices
        coords = np.empty((N,D))
        if 'C' == self.order:
            idx_order = xrange(D)
        else:
            idx_order = reversed(xrange(D))
        for d in idx_order:
            (coord,res) = divmod(res,self.coef[d])
            coords[:,d] = coord
        oob_mask = np.logical_or(indices < 0,
                                 indices >= self.physical_max_index)
        coords[oob_mask] = np.nan
        return coords.astype('i')

    def cell_shift(self):
        D = self.dim
        shift = np.empty(2**D)

        for (i,diff) in enumerate(itertools.product([0,1],repeat=D)):
            diff = np.array(diff)
            shift[i] = diff.dot(self.coef)
            
        return shift
        
