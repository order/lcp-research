from utils.parsers import KwargParser

import solvers
from solvers.value_iter import ValueIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import config.generator as gen
import time

class ValueIterGenerator(gen.SolverGenerator):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('termination_conditions',[])
        parser.add('recorders',[])
        parser.add_optional('notifications',[])
        args = parser.parse(kwargs)

        self.__dict__.update(args)
        
    def generate(self,discretizer):
        mdp_obj = discretizer.build_mdp()    
        iter = ValueIterator(mdp_obj)
        objects = {'mdp':mdp_obj}

        # Set up the solver object
        solver = solvers.IterativeSolver(iter)
        gen.add_trn(self,solver) # Termination, Recording, and Notify    

        return [solver,objects]

    def extract(self,solver,**kwargs):
        return gen.basic_extract(self,solver)

