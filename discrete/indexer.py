import numpy as np
import discretize

class Indexer(object):
    def __init__(self,lens,order='C'):
        lens = np.array(lens)
        (D,) = lens.shape
        self.dim = D
        
        self.max_index = np.product(lens)
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
        
    def coords_to_indices(self,coords):
        """
        Turn D dimensional coords into indices
        """
        (N,D) = coords.shape

        assert(discretize.is_int(coords)
               or 0 == np.sum(np.mod(coords,1.0)))
        
        assert(D == self.dim) # Dimension right
        assert(not np.any(coords < 0)) # all non
        assert(np.all(coords < self.lens))
        
        return coords.dot(self.coef)
    
    def indices_to_coords(self,indices):
        (N,) = indices.shape
        D = len(self.coef)
        
        assert(is_int(indices))     
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
                                 indices >= self.max_index)
        coords[oob_mask] = np.nan
        return coords.astype('i')
