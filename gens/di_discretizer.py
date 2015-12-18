import numpy as np

import mdp
from generator import Generator
from mdp.double_integrator import DoubleIntegratorRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class DIGenerator(Generator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('x_desc') # Mandatory
        parser.add('v_desc')
        parser.add('a_desc')
        
        parser.add('radius',0.25)
        parser.add('set_point',np.zeros(2))
        parser.add('oob_costs',1.0)
        parser.add('discount',0.99)
        
        parser.add('drain',False) # Flow 'drains; out of (0,0)
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        assert(0 < self.discount < 1)
       
    def generate(self):
        print "Generating discretizer..."
        start = time.time()
        xid,vid = 0,1

        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(self.x_desc,\
                                                             self.v_desc)
        physics = DoubleIntegratorRemapper()    
        cost_obj = mdp.BallCost(self.set_point,self.radius)
        actions = np.linspace(*self.a_desc)

        (x_lo,x_hi,x_n) = self.x_desc
        (v_lo,v_hi,v_n) = self.v_desc

        # (-inf,x_lo] out-of-bound node mapper
        left_oob_mapper = mdp.OOBSinkNodeMapper(xid,-float('inf'),
                                                x_lo,basic_mapper.num_nodes)
        # [x_hi,inf) out-of-bound node mapper
        right_oob_mapper = mdp.OOBSinkNodeMapper(xid,x_hi,float('inf'),
                                                 basic_mapper.num_nodes+1)

        # Map more extreme velocities back to edge
        state_remapper = mdp.RangeThreshStateRemapper(vid,v_lo,v_hi)

        discretizer = mdp.ContinuousMDPDiscretizer(physics,
                                                   basic_mapper,
                                                   cost_obj,
                                                   actions,
                                                   self.discount)
    
        discretizer.add_state_remapper(state_remapper)
        discretizer.add_node_mapper(left_oob_mapper)
        discretizer.add_node_mapper(right_oob_mapper)

        if self.drain:
            # Makes 0,0 a drain state (must be a pure state)
            discretizer.add_drain(np.array([0,0]))
        print "Built discretizer {0}s.".format(time.time() - start)

        return discretizer
