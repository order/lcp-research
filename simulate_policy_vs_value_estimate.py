import numpy as np
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt

from mdp.simulator import *

def compare_with_simulation(problem,
                            policy,
                            value_fn,
                            num_states,
                            horizon,
                            samples):
    start_states = problem.gen_model.boundary.random_points(
        num_states)

    V = value_fn.evaluate(start_states)

    R = np.zeros(num_states)
    for S in xrange(samples):
        (a,s,c) = simulate(problem,
                           policy,
                           start_states,
                           horizon)
        R += discounted_return(c,problem.discount)
    R /= float(samples)    
    return (V,R,start_states)

if __name__ == '__main__':
    parser = ArgumentParser(__file__,\
        'Form a policy from value function')
    parser.add_argument('val_fn_in_file',
                        metavar='FILE',
                        help='value function in file')
    parser.add_argument('policy_in_file',
                        metavar='FILE',
                        help='policy in file')
    parser.add_argument('problem_in_file',
                        metavar='FILE',
                        help='problem in file')
    parser.add_argument('data_out_file',
                        metavar='FILE',
                        help='data out file')
    args = parser.parse_args()

    FH = open(args.val_fn_in_file,'r')
    v_fn = pickle.load(FH) # primal/dual
    FH.close()

    FH = open(args.policy_in_file,'r')
    policy = pickle.load(FH)
    FH.close()


    FH = open(args.problem_in_file,'r')
    problem = pickle.load(FH)
    FH.close()    

    N = 2500 # Number of initial points
    H = 1000 # Rollout
    S = 1 # Number of samples
    (V,R,states) = compare_with_simulation(problem,
                                    policy,
                                    v_fn,
                                    N,H,S)

    FH = open(args.data_out_file,"w")
    pickle.dump((V,R,states),FH)
    FH.close()

