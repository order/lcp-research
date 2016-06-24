import numpy as np

from solvers import *

def solve_with_projective(plcp,thresh,max_iter,x0,y0,w0):
    iterator = ProjectiveIPIterator(plcp,x0=x0,y0=y0,w0=w0)
    solver = IterativeSolver(iterator)

    term_conds = [MaxIterTerminationCondition(max_iter)]
    announce = [IterAnnounce(),PotentialAnnounce(),PrimalDiffAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)

    solver.solve()
    return (iterator.get_primal_vector(),
            iterator.get_dual_vector())
