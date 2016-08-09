import numpy as np

from solvers import *
from lcp import *
def solve_lcp_with_extragradient(lcp,**kwargs):

    thresh   = kwargs.get('thresh',1e-12)
    max_iter = kwargs.get('max_iter',1000)
    
    (N,) = lcp.q.shape
    x0 = kwargs.get('x0',np.ones(N))

    iterator = ExtraGradientIterator(lcp,x0=x0,y0=y0)
    solver = IterativeSolver(iterator)

    term_conds = [PrimalChangeTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter)]
    announce = [IterAnnounce(),
                PotentialAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)    

    solver.solve()
    (p,d) = (iterator.get_primal_vector(),iterator.get_dual_vector())

    return (p,d,iterator.data)
