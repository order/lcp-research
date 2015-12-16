from utils.parsers import KwargParser

import solvers
from solvers.kojima import KojimaIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

from generator import SolverGenerator
import time

class KojimaGenerator(SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('discount',0.99)        
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        args = parser.parse(kwargs)

        # Dump into self namespace
        self.__dict__.update(args)
            
    def generate(self,discretizer):
        # Build objects
        mdp_obj = discretizer.build_mdp(discount=self.discount)
        lcp_obj = mdp_obj.tolcp()
        iter = KojimaIPIterator(lcp_obj)
        objects = {'mdp':mdp_obj,'lcp':lcp_obj}

        # Set up the solver object
        solver = solvers.IterativeSolver(iter)

        # Add termination conditions
        solver.termination_conditions.extend(
            self.termination_conditions.values())

        # Set up recorders
        self.recorder_names = self.recorders.keys()
        solver.recorders.extend(self.recorders.values())

        # Set up notification
        solver.notifications.extend(self.notifications.values())
    
        return [solver,objects]

    def extract(self,solver):               
        # Extract the value information
        # TODO: generalize
        names = self.recorder_names
        assert(len(names) == len(solver.recorders))
        
        data = {}
        for i in xrange(len(names)):
            data[names[i]] = np.array(solver.recorders[i].data)

        return data
    
