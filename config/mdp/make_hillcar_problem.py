import numpy as np
import utils
from mdp.transitions import HillcarTransitionFunction
from mdp.boundary import SaturationBoundary
from mdp.state_functions import BallSetFn
from mdp.costs import CostWrapper
from mdp.generative_model import GenerativeModel
from mdp.problem import Problem

from argparse import ArgumentParser
from utils.pickle import dump
import discrete
import matplotlib.pyplot as plt


def make_hillcar_problem(step_len,
                         n_steps,
                         jitter,
                         discount,
                         bounds,
                         goal,
                         cost_radius,
                         actions):
    """
    Makes a hillcar problem
    TODO: take in parameters
    """
    (A,action_dim) = actions.shape
    assert(action_dim == 1)

    state_dim = 2
    
    # Set up parameters for the DI problem
    trans_params = utils.kwargify(
        mass=1.0,
        step=step_len,
        num_steps=n_steps)
    
    trans_fn = HillcarTransitionFunction(
        **trans_params)
    
    boundary = SaturationBoundary(bounds)
    
    cost_state_fn = BallSetFn(goal, cost_radius)
    cost_fn = CostWrapper(cost_state_fn)
    cost_fn.favored=np.array([0.0])
    
    gen_model = GenerativeModel(trans_fn,
                                boundary,
                                cost_fn,
                                state_dim,
                                action_dim)

    action_boundary = [(actions[0],actions[-1])]

    problem = Problem(gen_model,
                      action_boundary,
                      discount)

    return problem
