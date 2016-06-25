import numpy as np

from solvers import *
from lcp import *
def solve_with_kojima(mdp,
                      disc,
                      **kwargs):

    thresh   = kwargs.get('thresh',1e-12)
    max_iter = kwargs.get('max_iter',1000)
    val_reg  = kwargs.get('val_reg',1e-15)
    flow_reg = kwargs.get('flow_reg',1e-12)

    # Build the LCP
    lcp_builder = LCPBuilder(mdp,
                             disc,
                             val_reg=val_reg,
                             flow_reg=flow_reg)
    #lcp_builder.remove_unreachable(0.0)
    lcp_builder.remove_oobs(0.0)
    lcp_builder.add_drain(np.zeros(2),0.0)
    lcp_builder.build_uniform_state_weights()
    lcp = lcp_builder.build()
    
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

    #pad_value = 1.0 / (1.0 - mdp.discount)
    pad_value = np.nan
    P = lcp_builder.expand(p,pad_value)
    D = lcp_builder.expand(d,pad_value)

    return (P,D,lcp_builder)
