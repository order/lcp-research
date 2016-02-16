import numpy as np

import mdp
from config import ProblemGenerator,DiscretizerGenerator
from mdp.problem import MDPProblem
from mdp.hallway_continuous import HallwayRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class HallwayGenerator(ProblemGenerator,DiscretizerGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('x_desc') # Mandatory
        parser.add('a_desc')
        
        parser.add('cost_obj')
        parser.add('discount')
        
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        assert(0 < self.discount < 1)
       
    def generate_problem(self):
        physics = HallwayRemapper()
        weight_obj = mdp.ConstFn(1.0) #Just use uniform

        xid=0
        (x_lo,x_hi,x_n) = self.x_desc
        boundary = [(x_lo,x_hi)]
        action_dim = 1

        # Map more extreme velocities back to edge
        state_remapper = mdp.RangeThreshStateRemapper(xid,x_lo,x_hi)

        problem = MDPProblem(physics,
                             boundary,
                             self.cost_obj,
                             weight_obj,
                             action_dim,
                             self.discount)
        problem.exception_state_remappers.append(state_remapper)
        self.problem = problem
        
        return problem

    def generate_discretizer(self):
        if not hasattr(self,problem):
            self.generate_problem()
        
        basic_mapper = mdp.InterpolatedRegularGridNodeMapper(self.x_desc)
        actions = np.linspace(*self.a_desc)

        discretizer = mdp.ContinuousMDPDiscretizer(self.problem,
                                                   basic_mapper,
                                                   actions)
        return discretizer
