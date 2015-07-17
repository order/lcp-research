import numpy as np
from math import pi,sqrt
import matplotlib.pyplot as plt
from mdp import MDP

import grid

import scipy.stats

# Norm pdf and derivative; move to another file?
def normpdf(x,mu,sigma):
    return scipy.stats.norm.pdf(x,loc=mu,scale=sigma)

def derv_normpdf(x,mu,sigma):
    """
    Derivative of the normal function
    """
    return (mu - x) / (sqrt(2*pi) * sigma**3) \
        * np.exp(-(x - mu)**2 / (2*sigma**2))

def physics_step(points,action,slope,**kwargs):
    """
    Physics step for hill car based on slope.
    """
    [N,d] = points.shape
    assert(d == 2)
    
    Mass = kwargs.get('mass',1)
    G = kwargs.get('gravity',9.806)
    step = kwargs.get('step',0.01)
    
    theta = np.arctan(slope(points[:,0]))
    F_grav = -Mass*G*np.sin(theta)
    F_cont = action*np.cos(theta)

    x_next = points[:,0] + step * points[:,1]
    v_next = points[:,1] + step * (F_grav + F_cont)
    return np.column_stack((x_next,v_next))
    
# Simple two hill terrain
def sample_hill_height(x):
    """
    A sample two hill terrain
    """
    a1 = 0.5
    mu1 = -1.25
    sigma1 = 0.25
    a2 = 1
    mu2 = 1.1
    sigma2 = 0.85
    return a1*normpdf(x,mu1,sigma1) + a2*normpdf(x,mu2,sigma2);
def sample_hill_slope(x):
    """
    A sample two hill terrain slope
    """
    a1 = 0.5
    mu1 = -1.25
    sigma1 = 0.25
    a2 = 1
    mu2 = 1.1
    sigma2 = 0.85
    return a1*derv_normpdf(x,mu1,sigma1) + a2*derv_normpdf(x,mu2,sigma2);

def plot_dynamics(points,action,slope):
    next_points = physics_step(points,action,slope)
    diff = next_points - points
    plt.quiver(points[:,0],points[:,1],diff[:,0],diff[:,1],\
               headwidth=1,headlength=2)
    plt.show()


def generate_hillcar_mdp(x_disc,v_disc,**kwargs):
    """
    Generate an MDP based on a regular grid discretization
    """
    slope_fn = kwargs.get('slope',sample_hill_slope)
    xlim = kwargs.get('xlim',1)
    vlim = kwargs.get('vlim',3)
    accel = kwargs.get('accel',1)

    Actions = [-accel,accel]

    # Generate the grid
    statespace = grid.Regular2DGrid(np.linspace(-xlim,xlim,x_disc),\
                                    np.linspace(-vlim,vlim,v_disc))
    N = statespace.n_points + 2
    left = statespace.n_points
    right = statespace.n_points+1
    
    Transitions = []
    Costs = []
    for action in Actions:
        next_points = physics_step(statespace.get_points(),action,slope_fn)

        # Velocity OOB dealt with by capping
        next_points[:,1] = np.minimum(next_points[:,1],vlim)
        next_points[:,1] = np.maximum(next_points[:,1],-vlim)
    
        (T,OOB) = statespace.points2matrix(next_points)

        # Deal with the x-coord out-of-bounds points
        T.resize((N,N)) 
        T[left,left] = 1.0 
        T[right,right] = 1.0
        for i in OOB:
            if next_points[i,0] > xlim:
                T[right,i] = 1.0
            else:
                assert(next_points[i,0] < -xlim)
                T[left,i] = 1.0
        c = np.ones(N)
        c[right] = 0

        # Use COO; may be using either T or T.T
        Transitions.append(scipy.sparse.coo_matrix(T))
        Costs.append(c)
    M = MDP(Transitions,Costs,Actions,name='Hillcar')
    return (M,statespace)
