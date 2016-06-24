import numpy as np

from solvers import *

def solve_with_kojima(mdp,thresh,max_iter,value_reg=1e-12,flow_reg=1e-12):
    lcp = mdp.build_lcp(value_reg,flow_reg)
    iterator = KojimaIPIterator(lcp)
    solver = IterativeSolver(iterator)

    term_conds = [PrimalChangeTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter)]
    announce = [IterAnnounce(),
                PotentialAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)

    solver.solve()
    return (iterator.get_primal_vector(),
            iterator.get_dual_vector())
