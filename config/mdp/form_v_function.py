import numpy as np
from argparse import ArgumentParser
from utils.pickle import dump, load

from mdp.state_functions import InterpolatedFunction
from mdp.solution_process import *

if __name__ == '__main__':
    parser = ArgumentParser(__file__,\
        'Form a policy from value function')
    parser.add_argument('sol_in_file',
                        metavar='FILE',
                        help='solution in file')
    parser.add_argument('disc_in_file',
                        metavar='FILE',
                        help='discretizer in file')
    parser.add_argument('mdp_in_file',
                        metavar='FILE',
                        help='mdp in file')
    parser.add_argument('v_fn_out_file',
                        metavar='FILE',
                        help='solution out file')
    args = parser.parse_args()

    p = load(args.sol_in_file)
    disc = load(args.disc_in_file)
    mdp = load(args.mdp_in_file)

    # Build the q policy
    (v,_) = split_solution(mdp,p)
    v_fn = InterpolatedFunction(disc,v)

    dump(v_fn,args.v_fn_out_file)
