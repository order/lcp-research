import numpy as np
from state_remapper import StateRemapper

class DoubleIntegratorRemapper(StateRemapper):
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.05)
        
    def remap(self,points,**kwargs):
        """
        Physics step for a double integrator:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """
        if 1 == len(points.shape):
            points = points[np.newaxis,:]
            
        [N,d] = points.shape
        assert(d == 2)
        
        assert('action' in kwargs)
        u = kwargs['action']
        
        x_next = points[:,0] + self.step * points[:,1]
        v_next = points[:,1] + self.step * u
        assert((N,) == x_next.shape)
        assert((N,) == v_next.shape)
        
        ret = np.column_stack([x_next,v_next])        
        assert((N,d) == ret.shape)
        
        return ret
