import numpy as np

from solvers import *
from lcp import *
def solve_with_kojima(lcp,**kwargs):

    thresh   = kwargs.get('thresh',1e-12)
    max_iter = kwargs.get('max_iter',1000)
    val_reg  = kwargs.get('val_reg',1e-15)
    flow_reg = kwargs.get('flow_reg',1e-12)

    iterator = KojimaIPIterator(lcp)
    solver = IterativeSolver(iterator)

    term_conds = [PotentialDiffTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter)]
    announce = [IterAnnounce(),
                PotentialAnnounce(),
                PotentialDiffAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)    

    solver.solve()
    (p,d) = (iterator.get_primal_vector(),iterator.get_dual_vector())

    return (p,d,iterator.data)
