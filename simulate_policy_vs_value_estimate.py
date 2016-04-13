import numpy as np
from argparse import ArgumentParser
import pickle

import matplotlib.pyplot as plt

from mdp.simulator import *

def compare_with_simulation(problem,
                            policy,
                            value_fn,
                            num_states,
                            horizon):
    start_states = problem.gen_model.boundary.random_points(
        num_states)

    V = value_fn.evaluate(start_states)

    (a,s,c) = simulate(problem,
                       policy,
                       start_states,
                       horizon)
    R = discounted_return(c,problem.discount)
    return (V,R)

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
    parser.add_argument('img_out_file',
                        metavar='FILE',
                        help='image out file')
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

    N = 250
    H = 1000
    (V,R) = compare_with_simulation(problem,
                                    policy,
                                    v_fn,
                                    N,
                                    H)
    assert((N,) == V.shape)
    assert((N,) == R.shape)
    l = min(np.min(V),np.min(R))
    u = max(np.max(V),np.max(R))
    plt.plot(V,R,'b.')
    plt.plot([l,u],[l,u],':r')
    plt.xlabel('Expected')
    plt.ylabel('Empirical')
    plt.show()
