import numpy as np
import utils
from mdp.transitions.double_integrator\
    import DoubleIntegratorTransitionFunction
from mdp.boundary import *
from mdp.state_functions import BallSetFn,TargetZoneFn
from mdp.costs import CostWrapper
from mdp.generative_model import GenerativeModel
from mdp.problem import Problem

from argparse import ArgumentParser
from utils.pickle import dump
import discrete
import matplotlib.pyplot as plt


def make_di_problem(step_len,
                    n_steps,
                    damp,
                    jitter,
                    discount,
                    bounds,
                    cost_radius,
                    actions):
    """
    Makes a double integrator problem
    TODO: take in parameters
    """
    (A,action_dim) = actions.shape
    assert(action_dim == 1)
    assert(actions[0] == -actions[-1])
    
    state_dim = 2
    
    # Set up parameters for the DI problem
    trans_params = utils.kwargify(step=step_len,
                                  num_steps=n_steps,
                                  dampening=damp,
                                  control_jitter=jitter)
    trans_fn = DoubleIntegratorTransitionFunction(
        **trans_params)
    
    #boundary = DoubleIntBoundary(bounds)
    boundary = SaturationBoundary(bounds)
    
    cost_state_fn = BallSetFn(np.zeros(2), cost_radius)
    #cost_state_fn = TargetZoneFn(np.array([[-0.5,0.5],[-0.5,0.5]]))
    cost_fn = CostWrapper(cost_state_fn)

    oob_costs = np.array([100]*2*state_dim)
    gen_model = GenerativeModel(trans_fn,
                                boundary,
                                cost_fn,
                                state_dim,
                                action_dim,
                                oob_costs)

    action_boundary = [(actions[0],actions[-1])]

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
