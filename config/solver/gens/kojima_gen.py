from utils.parsers import KwargParser

import solvers
from solvers.kojima import KojimaIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import config
import time

class KojimaGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('value_regularization')
        parser.add('flow_regularization')
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        args = parser.parse(kwargs)

        # Dump into self namespace
        self.__dict__.update(args)
            
    def generate(self,discretizer):
        # Build objects
        mdp_obj = discretizer.build_mdp()
        lcp_obj = mdp_obj\
            .build_lcp(value_regularization=self.value_regularization,
                       flow_regularization=self.flow_regularization)
        iter = KojimaIPIterator(lcp_obj)
        objects = {'mdp':mdp_obj,'lcp':lcp_obj}

        # Set up the solver object
        solver = solvers.IterativeSolver(iter)
        config.add_trn(self,solver) # Termination, Recording, and Notify      
    
        return [solver,objects]

    def extract(self,solver):               
        return config.basic_extract(self,solver)
    
