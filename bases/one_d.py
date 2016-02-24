import bases
import numpy as np
from utils.parsers import KwargParser

class ChebyshevBasis(bases.BasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('K')
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        
    def generate_basis(self,points,**kwargs):
        assert(len(points.shape) <= 2)
        if 2 == len(points.shape):
            assert(points.shape[1] == 1)
            points = points[:,0]

        x = points
                   
        (N,) = x.shape
        K = self.K
        assert(K >= 2)
        
        T = np.empty((N,K))
        T[:,0] = 1 # Const
        T[:,1] = x # Identity
        
        for i in xrange(2,K):
            T[:,i] = 2.0 * x * T[:,i-1] - T[:,i-2]

        return T
