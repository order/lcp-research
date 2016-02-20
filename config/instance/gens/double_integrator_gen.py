import numpy as np

import mdp
from config import ProblemGenerator,DiscretizerGenerator
from mdp import InterpolatedRegularGridNodeMapper as IRGNMapper
from mdp.problem import MDPProblem
from mdp.double_integrator import DoubleIntegratorRemapper
from utils.parsers import KwargParser

import time

#################################################
# Generate the DISCRETIZER object

class DoubleIntegratorGenerator(ProblemGenerator,DiscretizerGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('x_desc') # Mandatory
        parser.add('v_desc')
        parser.add('a_desc')
        parser.add('dampening')
        parser.add('step')
        
        parser.add('cost_obj')
        parser.add('discount')
        
        args = parser.parse(kwargs)
        
        self.__dict__.update(args)
        assert(0 < self.discount < 1)
       
    def generate_problem(self):
        physics = DoubleIntegratorRemapper(dampening=self.dampening,
                                           step=self.step)  
        weight_obj = mdp.ConstFn(1.0) #Just use uniform
        
        xid,vid = 0,1
        (x_lo,x_hi,x_n) = self.x_desc
        (v_lo,v_hi,v_n) = self.v_desc
        boundary = [(x_lo,x_hi),
                    (v_lo,v_hi)]

        action_dim = 1 # Have control over acceleration

        # Map more extreme velocities back to edge
        state_remapper = mdp.RangeThreshStateRemapper(vid,v_lo,v_hi)

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
        if not hasattr(self,'problem'):
            self.generate_problem()
        
        basic_mapper = IRGNMapper(self.x_desc,
                                  self.v_desc)
        actions = np.linspace(*self.a_desc)

        discretizer = mdp.ContinuousMDPDiscretizer(self.problem,
                                                   basic_mapper,
                                                   actions)
        xid = 0
        (x_lo,x_hi,x_n) = self.x_desc
        nnodes = basic_mapper.num_nodes

        # (-inf,x_lo] out-of-bound node mapper
        left_oob_mapper = mdp.OOBSinkNodeMapper(xid,-float('inf'),
                                                x_lo,
                                                nnodes)
        # [x_hi,inf) out-of-bound node mapper
        right_oob_mapper = mdp.OOBSinkNodeMapper(xid,x_hi,float('inf'),
                                                 nnodes+1)
        
        discretizer.add_node_mapper(left_oob_mapper)
        discretizer.add_node_mapper(right_oob_mapper)

        return discretizer
