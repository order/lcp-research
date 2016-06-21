import numpy as np
from utils.parsers import KwargParser

from transition import TransitionFunction

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
        h = self.step

        Samples = np.empty((samples,N,2))
        for s in xrange(samples):
            x = np.array(points[:,0])
            v = np.array(points[:,1])
            for i in xrange(self.num_steps):
                noise = np.random.randn(N);
                pert_acts = u + self.control_jitter * noise
                x += h * v + 0.5*h*h*pert_acts
                v *= (1.0 - damp)
                v += h * pert_acts
                
            Samples[s,:,0] = x
            Samples[s,:,1] = v
        return Samples
