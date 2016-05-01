import numpy as np
from argparse import ArgumentParser
import pickle

from mdp.policies import MinFunPolicy,IndexPolicyWrapper
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
    parser.add_argument('policy_out_file',
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
    q = q_vectors(mdp,v)
    q_fns = build_functions(mdp,disc,q)
    q_policy = IndexPolicyWrapper(MinFunPolicy(q_fns),
                                  mdp.actions)
    
    FH = open(args.policy_out_file,'w')
    pickle.dump(q_policy,FH)
    FH.close() 
