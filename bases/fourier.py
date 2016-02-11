import numpy as np
import bases
from utils.parsers import KwargParser
import utils

import numpy as np

def make_regular_frequencies(*Lens):
    # Generate all waves, then eliminate the all zero one
    linspaces = [np.linspace(0,2.0*np.pi,n) for n in Lens]
    W = utils.make_points(*linspaces)[1:,:]
    (K,D) = W.shape
    assert(D == 2)
    W = np.vstack([W,W])
    
    Phi = np.empty(2*K)
    Phi[:K] = 0
    Phi[K:] = np.pi / 2.0

    return (W.T,Phi)

class FourierBasis(bases.BasicBasisGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('frequencies')
        parser.add('shifts')
        args = parser.parse(kwargs)
        
        W = args['frequencies']
        assert(2 == len(W.shape))
        (D,K) = W.shape
        self.W = W
        self.K = K
        
        Phi = args['shifts']
        assert((K,) == Phi.shape)
        self.Phi = Phi        

    def isortho(self):
        return False
        
    def generate_basis(self,points,**kwargs):
        # Parse kwargs
        parser = KwargParser()
        parser.add('special_points',[])
        args = parser.parse(kwargs)
        special_points = args['special_points']
        
        (N,D) = points.shape
        K = self.K
        
        W = self.W
        Phi = self.Phi
        B = np.sqrt(2.0 / float(N)) * np.sin(points.dot(W) + Phi)
        assert((N,K) == B.shape)

        # Zero out any special points
        if len(special_points) > 0:
            B[special_points,:] = 0
                
        return B

class RandomFourierBasis(FourierBasis):

    def __init__(self,**kwargs):
        scale = kwargs.pop('scale',1.0) # Random scale
        K = kwargs.pop('num_basis',-1)
        D = kwargs.pop('dim',-1)
        assert(K >= 1)
        assert(D >= 1)
        
        W = scale * np.random.randn(D,K) # Weights
        Phi = 2.0 * np.pi * np.random.rand(K) # Phases shift
        
        super(RandomFourierBasis,self).__init__(frequencies = W,
                                                shifts = Phi,
                                                **kwargs)
