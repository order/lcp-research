import numpy as np
import bases
from utils.parsers import KwargParser

def gaussian_rbf(X,mu,bw):
    C = X - mu
    assert(C.shape == X.shape)
    RBF = np.exp(-np.power(np.linalg.norm(C,axis=1) / bw, 2))
    assert(RBF.shape[0] == X.shape[0])
    return RBF

class RadialBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('centers')
        parser.add('bandwidth')
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        
    def isortho(self):
        return False
        
    def generate_basis(self,points,**kwargs):
        # Parse kwargs
        parser = KwargParser()
        parser.add('special_points',[])
        args = parser.parse(kwargs)
        special_points = args['special_points']        

        (N,D) = points.shape
        (K,d) = self.centers.shape
        assert(d == D)

        B = []
        for i in xrange(K):
            mu = self.centers[i,:]
            assert((D,) == mu.shape)
            B.append(gaussian_rbf(points,mu,self.bandwidth))
        
        B = np.column_stack(B)
        assert((N,K) == B.shape)
        B[special_points,:] = 0
                
        return B
