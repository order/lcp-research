import numpy as np
import utils
from mdp.transitions import HillcarTransitionFunction
from mdp.boundary import *
from mdp.state_functions import BallSetFn
from mdp.costs import *
from mdp.generative_model import GenerativeModel
from mdp.problem import Problem

from argparse import ArgumentParser
from utils.pickle import dump
import discrete
import matplotlib.pyplot as plt


def make_hillcar_problem(step_len,
                         n_steps,
                         damp,
                         jitter,
                         discount,
                         bounds,
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
        num_steps=n_steps,
        dampening=damp,
        jitter=jitter)
    
    trans_fn = HillcarTransitionFunction(
        **trans_params)

    #boundary = SaturationBoundary(bounds)
    boundary = HillcarBoundary(bounds)
    
    cost_state_fn = BallSetFn(np.zeros(2), cost_radius)
    cost_fn = CostWrapper(cost_state_fn)
    cost_fn.favored=np.array([0.0])
    
    #If we see 100 leaking in, there is a problem with v saturation
    #with the boundary containment.
    oob_costs = np.array([0,0,0,0])
    # otherwise we will pay a penalty at the boundary, but be 0 after.
    
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
