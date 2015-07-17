import numpy as np
from mdp import MDP
import scipy.sparse

def generate_mdp(num_states,**kwargs):
    """ Generates an MDP based on the hallway problem
    Arguments: num_states (int)
    Returns: MDP object
    """
    # Wheel_slip: probability of going nowhere
    wheel_slip = kwargs.get('wheel_slipe',0.1)

    # Actions: move left or right
    Actions = [-1,1]

    # Form the cost
    target = num_states / 2
    cost = np.ones(num_states)
    cost[target] = 0
    Costs = [cost,cost]

    # Form the transition matrices
    Transitions = []
    for i in Actions:
        P = wheel_slip*scipy.sparse.eye(num_states) \
           + (1 - wheel_slip)*scipy.sparse.diags(np.ones(num_states-1),i)
        Transitions.append(P)

    return MDP(Transitions,Costs,Actions,name='Hallway')
    
print generate_mdp(5)
