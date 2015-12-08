from utils.parsers import KwargParser

import solvers
from solvers.mdp_split_ip import MDPSplitIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import numpy as np
import scipy as sp
import scipy.sparse as sps

from generator import SolverGenerator
import time

class MDPSplitIPGenerator(SolverGenerator):
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
        N = mdp_obj.num_states
        Phi = sps.eye(N)
        
        iter = MDPSplitIPIterator(mdp_obj,Phi,orthogonal=True)
        objects = {'mdp':mdp_obj,\
                   'basis':Phi,\
                   'proj_lcp':iter.proj_lcp_obj}


        # Set up the solver object
        solver = solvers.IterativeSolver(iter)

        # Add termination conditions
        max_iter_cond = MaxIterTerminationCondition(max_iter)
        val_change_term = ResidualTerminationCondition(thresh)
        solver.termination_conditions.append(val_change_term)

        # Set up recorders
        solver.recorders.append(PrimalRecorder())
        solver.recorders.append(DualRecorder())
    
        return [solver,objects]


    def extract(self,solver,**kwargs):
        assert(0 == len(kwargs))
               
        # Extract the value information
        # TODO: generalize
        names = ['primal','dual','steplen','primal_dir','dual_dir']
        assert(len(names) == len(solver.recorders))
        
        data = {}
        for i in xrange(len(names)):
            data[names[i]] = np.array(solver.recorders[i].data)

        return data
    
