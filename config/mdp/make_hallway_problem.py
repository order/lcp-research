import numpy as np
import utils
from mdp.transitions.hallway_discrete\
    import DiscreteHallwayTransition
from mdp.boundary import SaturationBoundary
from mdp.state_functions import BallSetFn
from mdp.costs import CostWrapper
from mdp.generative_model import GenerativeModel
from mdp.problem import Problem

from argparse import ArgumentParser
from utils.pickle import dump
import discrete
import matplotlib.pyplot as plt


def make_hallway_problem(N):
    """
    Makes a double integrator problem
    TODO: take in parameters
    """
    
    # Set up parameters for the DI problem
    stuck_p = 0.05
    trans_fn = DiscreteHallwayTransition(stuck_p,N)
    
    boundary = SaturationBoundary([(0,N-1)])
    
    cost_state_fn = BallSetFn(int(N/2), 0.25)
    cost_fn = CostWrapper(cost_state_fn)

    state_dim = 1
    action_dim = 1
    gen_model = GenerativeModel(trans_fn,
                                boundary,
                                cost_fn,
                                state_dim,
                                action_dim)

    action_boundary = [(-1,1)]
    discount = 0.95

    problem = Problem(gen_model,
                      action_boundary,
                      discount)

    return problem

if __name__ == '__main__':
    parser = ArgumentParser(__file__,
                            'Generates a double integrator problem')
    parser.add_argument('save_file',
                        metavar='FILE',
                        help='save file')
    args = parser.parse_args()
    
    problem = make_di_problem()
    dump(problem,args.save_file)
