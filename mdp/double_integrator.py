import numpy as np
import state_remapper

class DoubleIntegratorRemapper(state_remapper.StateRemapper):
    def __init__(self,step=0.01):
        self.step = step
        
    def remap(self,points,**kwargs):
        """
        Physics step for a double integrator:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """
        [N,d] = points.shape
        assert(d == 2)
        
        assert('action' in kwargs)
        u = kwargs['action']
        
        x_next = points[:,0] + self.step * points[:,1]
        v_next = points[:,1] + self.step * u
        return np.array((x_next,v_next)).T
