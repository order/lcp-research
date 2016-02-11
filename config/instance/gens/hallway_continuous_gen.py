import numpy as np

import mdp
from config import Generator
from mdp.hallway_continuous import HallwayRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class HallwayGenerator(Generator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('x_desc') # Mandatory
        parser.add('a_desc')
        
        parser.add('cost_obj')
        parser.add('discount')
        
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        assert(0 < self.discount < 1)
       
    def generate(self):
        print "Generating discretizer..."
        start = time.time()

        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(self.x_desc)
        physics = HallwayRemapper()    
        weight_obj = mdp.ConstFn(1.0) #Just use uniform
        actions = np.linspace(*self.a_desc)

        xid=0
        (x_lo,x_hi,x_n) = self.x_desc

        # Map more extreme velocities back to edge
        state_remapper = mdp.RangeThreshStateRemapper(xid,x_lo,x_hi)

        discretizer = mdp.ContinuousMDPDiscretizer(physics,
                                                   basic_mapper,
                                                   self.cost_obj,
                                                   weight_obj,
                                                   actions,
                                                   self.discount)
    
        discretizer.add_state_remapper(state_remapper)

        print "Built discretizer {0}s.".format(time.time() - start)

        return discretizer
