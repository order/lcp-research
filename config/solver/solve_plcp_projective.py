import numpy as np

from solvers import *

def solve_with_projective(plcp,**kwargs):
    x0 = kwargs['x0']
    y0 = kwargs['y0']
    w0 = kwargs['w0']
    max_iter = kwargs.get('max_iter',1000)
    thresh = kwargs.get('thresh',1e-12)
    
    iterator = ProjectiveIPIterator(plcp,
                                    x0=x0,
                                    y0=y0,
                                    w0=w0)
    solver = IterativeSolver(iterator)

    # No threshold condition yet
    
    term_conds = [InnerProductTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter),
                  SteplenTerminationCondition(1e-20)]
    
    announce = [IterAnnounce(),PotentialAnnounce(),PrimalDiffAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)

    solver.solve()
    return (iterator.get_primal_vector(),
            iterator.get_dual_vector(),
            iterator.data)
