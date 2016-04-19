import numpy as np
from argparse import ArgumentParser
import pickle

from mdp.policy import MaxFunPolicy,IndexPolicyWrapper,RandomDiscretePolicy
from mcts.mcts_policy import MCTSPolicy
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
    parser.add_argument('problem_in_file',
                        metavar='FILE',
                        help='problem in file')
    parser.add_argument('policy_out_file',
                        metavar='FILE',
                        help='solution out file')
    parser.add_argument('-z','--horizon',
                        default=25,
                        type=int,
                        help='rollout horizon')
    parser.add_argument('-b','--budget',
                        default=100,
                        type=int,
                        help='expansion budget')
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

    FH = open(args.problem_in_file,'r')
    problem = pickle.load(FH)
    FH.close()  

    # Build the q policy
    (v,flow) = split_solution(mdp,p)
    flow_fns = build_functions(mdp,disc,flow)
    flow_policy = MaxFunPolicy(flow_fns)
    #rand_policy = RandomDiscretePolicy(np.array([0.25,0.5,0.25]))
    v_fn = InterpolatedFunction(disc,v)

    mcts_policy = MCTSPolicy(problem,
                             mdp.actions,
                             flow_policy, # rollout policy
                             v_fn, # fathom estimate
                             args.horizon, # rollout horizon
                             args.budget) # expansion budget
    
    FH = open(args.policy_out_file,'w')
    pickle.dump(mcts_policy,FH)
    FH.close() 
