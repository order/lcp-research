from utils.parsers import KwargParser

import solvers
from solvers.value_iter import ValueIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

from generator import SolverGenerator
import time

class ValueIterGenerator(SolverGenerator):
    def generate(self,**kwargs):
        parser = KwargParser()
        parser.add('discretizer')
        parser.add('discount',0.99)
        parser.add('max_iter',1000)
        parser.add('thresh',1e-6)
        args = parser.parse(kwargs)

        discretizer = args['discretizer']
        discount = args['discount']
        max_iter = args['max_iter']
        thresh = args['thresh']

        mdp_obj = discretizer.build_mdp(discount=discount)    
        iter = ValueIterator(mdp_obj)
        objects = {'mdp':mdp_obj}


        # Set up the solver object
        solver = solvers.IterativeSolver(iter)

        # Add termination conditions
        max_iter_cond = MaxIterTerminationCondition(max_iter)
        val_change_term = ValueChangeTerminationCondition(thresh)
        solver.termination_conditions.append(val_change_term)
        solver.termination_conditions.append(max_iter_cond)


        # Set up notifications
        #solver.notifications.append(ValueChangeAnnounce())

        # Set up recorders
        solver.recorders.append(ValueRecorder())

        return [solver,objects]


    def extract(self,solver,**kwargs):
        # Extract the value information
        values = np.array(solver.recorders[0].data)
        
        mdp_obj = solver.iterator.mdp
        n = mdp_obj.num_states
        A = mdp_obj.num_actions    
        N = n*(A+1)

        assert(n == values.shape[1])
        I = values.shape[0]
    
        primal = np.empty((I,N))
        primal[:,n:] = np.nan
        primal[:,:n] = values

        data = {'primal':primal}
    
        return data
    
