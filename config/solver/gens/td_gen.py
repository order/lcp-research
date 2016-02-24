from utils.parsers import KwargParser
import utils

import solvers
from solvers.td import TabularTDIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import config
import time

class TabularTDGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('termination_conditions',[])
        parser.add('recorders',[])
        parser.add_optional('notifications',[])
        parser.add('num_samples')
        parser.add('step_size')
        parser.add('policy')
        args = parser.parse(kwargs)

        self.__dict__.update(args)
        
    def generate(self,discretizer):
        mdp_obj = discretizer.build_mdp()
        points = discretizer.get_node_states()
        objects = {'mdp':mdp_obj}

        # Set up the solver object
        policy_vector = self.policy.get_action_indices(points)
        options = utils.kwargify(num_samples=self.num_samples,
                                 step_size=self.step_size,
                                 policy = policy_vector)
        iter = TabularTDIterator(mdp_obj,**options)
        
        solver = solvers.IterativeSolver(iter)
        config.add_trn(self,solver) # Termination, Recording, and Notify    

        return [solver,objects]

    def extract(self,solver,**kwargs):
        return config.basic_extract(self,solver)

