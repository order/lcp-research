import numpy as np
import scipy.sparse as sps
from mdp import *
from config.mdp import *

from discrete import make_points

import matplotlib.pyplot as plt

def build_basis(disc):

    N = disc.num_nodes()
    n = disc.num_real_nodes()
    
    points = np.empty((N,2))
    
    points[:n,:] = disc.get_cutpoints()
    points[n:,:] = np.zeros((N-n,2))

    K = 5
    W = make_points([np.linspace(0,2.0*np.pi/10,K)]*2)

    B = []
    B.append(np.cos(W.dot(points.T)))
    B.append(np.sin(W[1:,:].dot(points.T)))

    B = np.vstack(B).T
    c = np.random.randn(49)
    print c
    Z = B[:n,:].dot(c)
    plt.imshow(Z.reshape((21,21)),interpolation='none')
    
    plt.show()

def build_problem():
    disc_n = 20 # Number of cells per dimension
    step_len = 0.01           # Step length
    n_steps = 5               # Steps per iteration
    damp = 0.01               # Dampening
    jitter = 0.1              # Control jitter 
    discount = 0.99           # Discount (\gamma)
    B = 5
    bounds = [[-B,B],[-B,B]]  # Square bounds, 
    cost_radius = 0.25        # Goal region radius
    
    actions = np.array([[-1],[0],[1]]) # Actions
    action_n = 3
    assert(actions.shape[0] == action_n)

    # Uniform start states
    problem = make_di_problem(step_len,
                              n_steps,
                              damp,
                              jitter,
                              discount,
                              bounds,
                              cost_radius,
                              actions)
    # Generate MDP
    (mdp,disc) = make_uniform_mdp(problem,disc_n,action_n)
    add_drain(mdp,disc,np.zeros(2),0)
    return (mdp,disc)

(mdp,disc) = build_problem()
basis = build_basis(disc)
