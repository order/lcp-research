from utils.kwargparser import KwargParser

import solvers
from solvers.kojima import KojimaIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *

import time


def build_solver(discretizer,**kwargs):
    parser = KwargParser()
    parser.add('discount',0.99)
    parser.add('max_iter',1000)
    parser.add('thresh',1e-6)
    args = parser.parse(kwargs)
    
    discount = args['discount']
    max_iter = args['max_iter']
    thresh = args['thresh']

    mdp_obj = discretizer.build_mdp(discount=discount)
    lcp_obj = mdp_obj.tolcp()
    iter = KojimaIPIterator(lcp_obj)
    objects = {'mdp':mdp_obj,'lcp':lcp_obj}


    # Set up the solver object
    solver = solvers.IterativeSolver(iter)

    # Add termination conditions
    max_iter_cond = MaxIterTerminationCondition(max_iter)
    val_change_term = ResidualTerminationCondition(thresh)
    solver.termination_conditions.append(val_change_term)

    # Set up recorders
    solver.recorders.append(PrimalRecorder())
    solver.recorders.append(DualRecorder())
    solver.recorders.append(StepLenRecorder())
    solver.recorders.append(PrimalDirRecorder())
    solver.recorders.append(DualDirRecorder())
    
    return [solver,objects]


def extract_data(solver):
    # Extract the value information

    # TODO: generalize
    names = ['primal','dual','steplen','primal_dir','dual_dir']
    assert(len(names) == len(solver.recorders))

    data = {}
    for i in xrange(len(names)):
        data[names[i]] = np.array(solver.recorders[i].data)

    return data
    
