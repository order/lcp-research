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

    vals = value_fn.evaluate(start_states)

    returns = np.zeros(num_states)
    actions = []
    states = []
    costs = []
    for S in xrange(samples):
        (a,s,c) = simulate(problem,
                           policy,
                           start_states,
                           horizon)
        actions.append(a)
        states.append(s)
        costs.append(c)
        returns += discounted_return(c,problem.discount)
    returns /= float(samples)

    
    return (vals,returns,start_states,actions, states, costs)

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
    parser.add_argument('-s','--samples',
                        metavar='S',
                        help='number of samples',
                        default=250,
                        type=int)
    parser.add_argument('-r','--rollout',
                        metavar='R',
                        help='rollout length',
                        default=500,
                        type=int)
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

    N = args.samples # Number of initial points
    H = args.rollout # Rollout
    S = 1 # Number of samples
    ret = compare_with_simulation(problem,
                                  policy,
                                  v_fn,
                                  N,H,S)

    FH = open(args.data_out_file,"w")
    pickle.dump(ret,FH)
    FH.close()

