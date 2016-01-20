import numpy as np

import mdp
from config.generator import Generator
from mdp.hillcar import HillcarRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class HillcarGenerator(Generator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('x_desc')
        parser.add('v_desc')
        parser.add('a_desc')
        parser.add('cost_obj')
        parser.add('discount')
        parser.add('slope')
        args = parser.parse(kwargs)
        self.__dict__.update(args)

        assert(0 < self.discount < 1)

    def generate(self):
        xid,vid = 0,1

        print "Generating discretizer..."
        start = time.time()
        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(self.x_desc,
                                                             self.v_desc)
        physics = HillcarRemapper(slope=self.slope)
        weight_obj = mdp.ConstFn(1.0) #Just use uniform
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
                                                   self.cost_obj,
                                                   weight_obj,
                                                   actions,
                                                   self.discount)
    
        discretizer.add_state_remapper(state_remapper)
        discretizer.add_node_mapper(left_oob_mapper)
        discretizer.add_node_mapper(right_oob_mapper)
        print "Built discretizer {0}s.".format(time.time() - start)

        return discretizer
