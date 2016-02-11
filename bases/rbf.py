import numpy as np
from scipy.spatial.distance import cdist
import bases
from utils.parsers import KwargParser

def mahalanobis_dist(X,mu,Cov):
    (N,d) = X.shape

    if not isinstance(mu,np.ndarray):
        mu = np.array([[mu]])
        
    if 1 == len(mu.shape):
        assert(d == mu.size)
        mu = mu[np.newaxis,:]
    else:
        assert((1,d) == mu.shape)

    # Calculate the Mahalanobis distances
    if isinstance(Cov,np.ndarray):
        VI = np.linalg.inv(Cov)
    else:
        VI = 1.0 / Cov        
    dist = cdist(X,mu,'mahalanobis',VI=VI).squeeze()
    assert((N,) == dist.shape)
    return dist

def laplacian_rbf(X,mu,Cov):
    (N,d) = X.shape
    dist = mahalanobis_dist(X,mu,Cov)
    
    # Get the scaling factor
    # This is the wrong scaling factor
    if isinstance(Cov,np.ndarray):
        D = np.linalg.det(Cov)
    else:
        D = Cov
    scale = 1.0 / np.sqrt( np.power(2.0*np.pi,d) * D)

    return scale * np.exp( -0.5 * dist)

def gaussian_rbf(X,mu,Cov):
    (N,d) = X.shape
    dist = mahalanobis_dist(X,mu,Cov)

    # Get the scaling factor
    if isinstance(Cov,np.ndarray):
        D = np.linalg.det(Cov)
    else:
        D = Cov
    scale = 1.0 / np.sqrt( np.power(2.0*np.pi,d) * D)

    # Return the normalized Gaussian
    return scale * np.exp( -0.5 * np.power(dist,2.0))

class RadialBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('centers')
        parser.add('covariance')
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        
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
        if len(special_points) > 0:
            B[special_points,:] = 0
                
        return B
