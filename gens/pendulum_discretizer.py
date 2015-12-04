import numpy as np

import mdp
from mdp.discretizer import DiscretizerGenerator
from mdp.pendulum import PendulumRemapper
from mdp.costs import TargetZoneCost
from mdp.state_remapper import AngleWrapStateRemaper,RangeThreshStateRemapper
from utils.kwargparser import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class PendulumGenerator(DiscretizerGenerator):
    def generate(self,**kwargs):
        print "Generating discretizer..."
        start = time.time()
        xid,vid = 0,1

        parser = KwargParser()
        parser.add('q_n')
        parser.add('dq_desc')
        parser.add('a_desc')
        parser.add('nudge',np.pi / 16.0)
        parser.add('set_point',np.array([np.pi,0.0]))
        parser.add('oob_costs',1.0)
        parser.add('discount',0.99)
        args = parser.parse(kwargs)

        q_n = args['q_n']
        dq_desc = args['dq_desc']
        a_desc = args['a_desc']
        nudge = args['nudge']
        set_point = args['set_point']
        oob_cost = args['oob_costs']
        discount = args['discount']
        assert(0 < discount < 1)

        q_desc = (0.0,2*np.pi,q_n)

        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(q_desc,dq_desc)
        physics = PendulumRemapper()    
        cost_obj = TargetZoneCost(np.array([\
            [np.pi-nudge,np.pi+nudge],\
            [-nudge,nudge]]))        
        actions = np.linspace(*a_desc)

        # Angles wrap around
        q_angle_wrap = AngleWrapStateRemaper(0)
        dq_thresh = RangeThreshStateRemapper(1,dq_desc[0],dq_desc[1])

        discretizer = mdp.ContinuousMDPDiscretizer(physics,
                                                   basic_mapper,
                                                   cost_obj,actions)
    
        discretizer.add_state_remapper(q_angle_wrap)
        discretizer.add_state_remapper(dq_thresh)
        print "Built discretizer {0}s.".format(time.time() - start)

        return discretizer
