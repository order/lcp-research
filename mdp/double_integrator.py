import numpy as np
from math import pi,sqrt
import matplotlib.pyplot as plt
import mdp
import state_remapper

import grid

import scipy.stats

class DoubleIntegratorRemapper(state_remapper.StateRemapper):
    def __init__(self,step=0.01):
        self.step = step
        
    def remap(self,points,**kwargs):
        """
        Physics step for a double integrator:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """
        [N,d] = points.shape
        assert(d == 2)
        
        assert('action' in kwargs)
        u = kwargs['action']
        
        x_next = points[:,0] + self.step * points[:,1]
        v_next = points[:,1] + self.step * u
        return np.array((x_next,v_next)).T
        
def plot_trajectory(policy_fn,**kwargs):
    x = np.rand(1,2)
    I = 1000
    X = np.zeros(I,2)
    for i in xrange(I):
        x = physics_step(x,policy_fn(x))
        X[i,:] = x
    plt.plot(X[:,0],X[:,1])
    plt.show()

def generate_mdp(x_disc,v_disc,**kwargs):
    """
    Generate an MDP based on a regular grid discretization
    """
    slope_fn = kwargs.get('slope',sample_hill_slope)
    xlim = kwargs.get('xlim',1)
    vlim = kwargs.get('vlim',3)
    accel = kwargs.get('accel',1)
    assert(accel > 0)
    num_actions = kwargs.get('num_actions',2)

    Actions = np.linspace(-accel,accel,num=num_actions)

    # Generate the grid
    statespace = grid.Regular2DGrid(np.linspace(-xlim,xlim,x_disc),\
                                    np.linspace(-vlim,vlim,v_disc))
    N = statespace.n_points + 1
    oob_state = statespace.n_points # out-of-bounds state
    
    Transitions = []
    Costs = []
    for action in Actions:
        next_points = physics_step(statespace.get_points(),action,slope_fn)

        # Velocity OOB dealt with by capping    
        (T,OOB) = statespace.points2matrix(next_points)

        # Deal with the x-coord out-of-bounds points
        T.resize((N,N)) 
        T[oob_state,oob_state] = 1.0 
        for i in OOB:
            T[oob_state,i] = 1.0 # Transition to oob state.
        c = np.ones(N)
        c[oob_state] = 1.0

        # Use COO; may be using either T or T.T
        Transitions.append(scipy.sparse.coo_matrix(T))
        Costs.append(c)
    M = mdp.MDP(Transitions,Costs,Actions,name='Double Integrator')
    return (M,statespace)
