import numpy as np
import math
from utils.parsers import KwargParser
from transition import TransitionFunction

class HillcarTransitionFunction(TransitionFunction):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('mass')
        parser.add('step')
        parser.add('num_steps')
        parser.add('jitter')
        args = parser.parse(kwargs)
        
        self.g = 9.806
        self.m = 1

        self.__dict__.update(args)
        
    def multisample_transition(self,points,actions,samples=1):
        """
        Physics step for hill car based on slope.
        """
        [N,d] = points.shape
        assert(d == 2)

        # Handling different action inputs
        if isinstance(actions,np.ndarray):
            if 2 == len(actions.shape):
                assert((N,1) == actions.shape)
                u = actions[:,0]
            else:
                assert((1,) == actions.shape)
                u = actions[0]
        else:
            assert(type(actions) in [int,float])
            u = actions
        
        t = self.step
        n_step = self.num_steps

        res = np.empty((samples,N,d))
        for s in xrange(samples):
            curr = np.array(points)
            for _ in xrange(n_step):
                noise = self.jitter * np.random.randn(*u.shape)
                q = new_slope(curr[:,0])
                p = 1 + (q * q)
                a = ((u+noise) / (self.m * np.sqrt(p)))\
                    -(self.g * q / p)
            
                curr[:,0] += t * points[:,1] + 0.5*t*t*a
                curr[:,1] += t * a
            res[s,:,:] = curr
        return res


def classic_hill(x):
    A = 1.0
    B = 5.0
    C = 0.0
    def f1(x):
        return x * (x + 1)
    def f2(x):
        return (A * x) / np.sqrt(1 + B * x * x)
        
    (N,) = x.shape
    ret = np.empty(N)
    mask = x < C
    ret[mask] = f1(x[mask])
    ret[~mask] = f2(x[~mask])
    return ret
    
def classic_slope(x):
    A = 1.0
    B = 5.0
    C = 0.0
    def f1_dashed2(x):
        return (2.0 * x + 1.0)
    def f2_dashed2(x):
        a = np.sqrt(1 + B * x * x)
        return A / (a * a * a)
    (N,) = x.shape
    ret = np.empty(N)
    mask = x < C
    ret[mask] = f1_dashed2(x[mask])
    ret[~mask] = f2_dashed2(x[~mask])
    return ret

def triangle_wave(x,a,s):
    x = x/a
    return s*(2 * np.abs(2* (x - np.floor(x+0.5))) - 1)

def new_slope(x):
    a = 8
    th = 0.05
    s = 1
    
    sig = triangle_wave(x - a/4,a,s)
    sig = np.sign(sig) * np.maximum(0,np.abs(sig)-th)
    return sig

