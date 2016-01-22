import numpy as np
import bases

class RandomFourierBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        self.scale = kwargs.get('scale',1.0)
        self.K = kwargs['num_basis']
        assert(self.K >=1)
        
    def generate(self,points,**kwargs):
        K = self.K
        (N,D) = points.shape
        special_points = kwargs.get('special_points',[])

        W = self.scale * np.random.randn(D,K-1) # Weights
        Phi = 2.0 * np.pi * np.random.rand(K-1) # Phases shift
        F = np.sin(points.dot(W) + Phi) # Non-constant columns
        assert((N,K-1) == F.shape)

        # Format: B = [1 | fourier columns]        
        B = np.hstack([1/np.sqrt(N)*np.ones((N,1)),
                       np.sqrt(2.0 / float(N)) * F])
        assert((N,K) == B.shape)

        # Zero out any special points
        if len(special_points) > 0:
            B[special_points,:] = 0
                
        return B
