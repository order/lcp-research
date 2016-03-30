import numpy as np
from state_remapper import StateRemapper

class HallwayRemapper(StateRemapper):
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.05)
        
    def remap(self,points,**kwargs):
        """
        Physics step for hallway:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """            
        (N,d) = points.shape
        assert(d == 1)
        
        assert('action' in kwargs)
        u = kwargs['action']
        
        x_next = points + self.step * u
        assert((N,1) == x_next.shape)
                
        return x_next
