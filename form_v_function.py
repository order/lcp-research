import numpy as np
from argparse import ArgumentParser
import pickle

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

    FH = open(args.sol_in_file,'r')
    (p,d) = pickle.load(FH) # primal/dual
    FH.close()

    FH = open(args.disc_in_file,'r')
    disc = pickle.load(FH)
    FH.close()

    FH = open(args.mdp_in_file,'r')
    mdp = pickle.load(FH)
    FH.close()    

    # Build the q policy
    (v,_) = split_solution(mdp,p)
    v_fn = InterpolatedFunction(disc,v)
    
    FH = open(args.v_fn_out_file,'w')
    pickle.dump(v_fn,FH)
    FH.close() 
