import numpy as np
from argparse import ArgumentParser
from utils.pickle import dump, load

from solvers.value_iter import ValueIterator
from solvers import *

def solve_with_value_iter(mdp,thresh,max_iter):
    iterator = ValueIterator(mdp)
    solver = IterativeSolver(iterator)

    term_conds = [ValueChangeTerminationCondition(thresh),
                  MaxIterTerminationCondition(max_iter)]
    announce = [ValueChangeAnnounce()]
    solver.termination_conditions.extend(term_conds)
    solver.notifications.extend(announce)

    solver.solve()
    return iterator.get_value_vector()

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

    mdp = load(args.mdp_in_file)
    sol = solve_with_kojima(mdp,1e-9,1e4)
    dump(p, args.sol_out_file)
