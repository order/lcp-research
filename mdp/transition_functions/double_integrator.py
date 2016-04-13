import numpy as np
from utils.parsers import KwargParser

from mdp.transition import TransitionFunction

class DoubleIntegratorTransitionFunction(TransitionFunction):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('step')
        parser.add('num_steps')
        parser.add('dampening')
        parser.add('control_jitter')
        args = parser.parse(kwargs)

        self.__dict__.update(args)
        
    def multisample_transition(self,points,actions,samples=1):
        """
        Physics step for a double integrator:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """
        (N,d) = points.shape
        assert(d == 2) # Generalize

        # Handling different action inputs
        if isinstance(actions,np.ndarray):
            if 2 == len(actions.shape):
                assert((N,1) == actions.shape)
                actions = actions[:,0]
            else:
                assert((1,) == actions.shape)
        else:
            assert(type(actions) in [int,float])
        
        u = actions # Could be either a num, or vector
        damp = self.dampening

        Samples = np.empty((samples,N,2))
        T = np.array([[1,self.step],[0,(1-damp)]])
        for s in xrange(samples):
            curr = points
            for i in xrange(self.num_steps):
                curr = points.dot(T.T)
                noise = self.control_jitter*np.random.randn(N)
                curr[:,1] += self.step * (u + noise)
            Samples[s,:,:] = curr        
        return Samples
