import numpy as np

import mdp
from config.generator import Generator
from mdp.pendulum import PendulumRemapper
from mdp.costs import TargetZoneCost
from mdp.state_remapper import AngleWrapStateRemaper,RangeThreshStateRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class PendulumGenerator(Generator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('q_n')
        parser.add('dq_desc')
        parser.add('a_desc')
        parser.add('cost_obj')
        parser.add('discount',0.99)
        args = parser.parse(kwargs)

        self.__dict__.update(args)
        assert(0.0 < self.discount < 1.0)
        
    def generate(self,**kwargs):
        print "Generating discretizer..."
        start = time.time()
        xid,vid = 0,1

        q_desc = (0.0,2*np.pi,self.q_n)

        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(q_desc,
                                                             self.dq_desc)
        physics = PendulumRemapper()
        weight_obj = mdp.ConstFn(1.0) #Just use uniform
        actions = np.linspace(*self.a_desc)
        
        # Angles wrap around
        q_angle_wrap = AngleWrapStateRemaper(xid)
        dq_thresh = RangeThreshStateRemapper(vid,
                                             self.dq_desc[0],
                                             self.dq_desc[1])

        discretizer = mdp.ContinuousMDPDiscretizer(physics,
                                                   basic_mapper,
                                                   self.cost_obj,
                                                   weight_obj,
                                                   actions,
                                                   self.discount)
    
        discretizer.add_state_remapper(q_angle_wrap)
        discretizer.add_state_remapper(dq_thresh)
        print "Built discretizer {0}s.".format(time.time() - start)

        return discretizer
