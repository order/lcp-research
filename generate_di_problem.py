import numpy as np
import utils
from mdp.transition_functions.double_integrator\
    import DoubleIntegratorTransitionFunction
from mdp.boundary import SaturationBoundary
from mdp.state_functions import BallSetFn,TargetZoneFn
from mdp.costs import CostWrapper
from mdp.generative_model import GenerativeModel
from mdp.problem import Problem

from argparse import ArgumentParser
import pickle
import discrete
import matplotlib.pyplot as plt


def make_di_problem():
    state_dim = 2
    action_dim = 1
    
    # Set up parameters for the DI problem
    trans_params = utils.kwargify(step=0.01,
                                  num_steps=5,
                                  dampening=0.01,
                                  control_jitter=0.01)
    trans_fn = DoubleIntegratorTransitionFunction(
        **trans_params)
    
    boundary = SaturationBoundary([(-6,6),(-5,5)])
    
    #cost_state_fn = BallSetFn(np.zeros(2), 0.5)
    cost_state_fn = TargetZoneFn(np.array([[-0.5,0.5],[-0.5,0.5]]))
    cost_fn = CostWrapper(cost_state_fn)
    
    gen_model = GenerativeModel(trans_fn,
                                boundary,
                                cost_fn,
                                state_dim,
                                action_dim)

    action_boundary = [(-1,1)]
    discount = 0.997

    problem = Problem(gen_model,
                      action_boundary,
                      discount)

    return problem

if __name__ == '__main__':
    parser = ArgumentParser(__file__,'Generates a continuous double integrator problem')
    parser.add_argument('save_file',
                        metavar='FILE',
                        help='save file')
    args = parser.parse_args()
    
    problem = make_di_problem()
    FH = open(args.save_file,'w')
    pickle.dump(problem,FH)
    FH.close()
