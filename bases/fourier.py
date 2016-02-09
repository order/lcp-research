import numpy as np
import bases

class FourierBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('scale')
        parser.add('num_basis')
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
        K = self.num_basis
        
        W = self.scale * np.random.randn(D,K) # Weights
        Phi = 2.0 * np.pi * np.random.rand(K) # Phases shift
        B = np.sqrt(2.0 / float(N)) * np.sin(points.dot(W) + Phi)
        assert((N,K) == B.shape)

        # Zero out any special points
        if len(special_points) > 0:
            B[special_points,:] = 0
                
        return B
