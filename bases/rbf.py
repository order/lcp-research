import numpy as np
from scipy.spatial.distance import cdist
import bases
from utils.parsers import KwargParser

def gaussian_rbf(X,mu,Cov):
    (N,d) = X.shape

    if 1 == len(mu.shape):
        assert(d == mu.size)
        mu = mu[np.newaxis,:]
    else:
        assert((1,d) == mu.shape)

    # Calculate the Mahalanobis distances
    VI = np.linalg.inv(Cov)
    dist = cdist(X,mu,'mahalanobis',VI=VI).squeeze()
    assert((N,) == dist.shape)

    # Get the scaling factor
    D = np.linalg.det(Cov)
    scale = 1.0 / np.sqrt( np.power(2.0*np.pi,d) * D)

    # Return the normalized Gaussian
    return scale * np.exp( -0.5 * dist)

class RadialBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('centers')
        parser.add('covariance')
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
            B.append(gaussian_rbf(points,mu,self.covariance))
        
        B = np.column_stack(B)
        assert((N,K) == B.shape)
        B[special_points,:] = 0
                
        return B
