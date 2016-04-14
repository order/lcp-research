import numpy as np
from argparse import ArgumentParser
import pickle

from solvers.kojima import KojimaIPIterator
from solvers import IterativeSolver,\
    PrimalChangeTerminationCondition,\
    MaxIterTerminationCondition,\
    PrimalChangeAnnounce

def solve_with_kojima(mdp,thresh,max_iter):
    lcp = mdp.build_lcp(1e-12,1e-8)
    iterator = KojimaIPIterator(lcp)
    solver = IterativeSolver(iterator)

    term_conds = [PrimalChangeTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter)]
    announce = [PrimalChangeAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)

    solver.solve()
    return (iterator.get_primal_vector(),
            iterator.get_dual_vector())

if __name__ == '__main__':
    parser = ArgumentParser(__file__,\
        'Generates an MDP from continuous problem')
    parser.add_argument('mdp_in_file',
                        metavar='FILE',
                        help='mdp file')
    parser.add_argument('sol_out_file',
                        metavar='FILE',
                        help='solution out file')
    args = parser.parse_args()

    FH = open(args.mdp_in_file,'r')
    mdp = pickle.load(FH)
    FH.close()

    """
    ...
    """
    sol = solve_with_kojima(mdp,1e-9,1e4)
    
    FH = open(args.sol_out_file,'w')
    pickle.dump(sol,FH)
    FH.close() 
