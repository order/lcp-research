import numpy as np
import matplotlib.pyplot as plt

import discrete.regular_interpolate as reg_interp
from discrete.discretize import make_points

import mdp.transition_functions.double_integrator as di
import mdp.transition_functions.mdp_wrapper as mdp_wrapper
import mdp.state_remapper as remapper
import mdp.state_functions as state_fn
import mdp.costs as costs
import mdp.mdp_builder as mdp_builder
import solvers
from solvers.value_iter import ValueIterator
import utils


# Transition function
di_params = utils.kwargify(step=0.01,
                           num_steps=5,
                           dampening=0.01,
                           control_jitter=0)
transition_function = di.DoubleIntegratorTransitionFunction(**di_params)

# Discretizer
X = 74
Y = 74
x_grid_desc = (-5,5,X)
v_grid_desc = (-5,5,Y)
grid_descs = [x_grid_desc,v_grid_desc]
discretizer = reg_interp.RegularGridInterpolator(grid_descs)


# State_remappers
state_remappers = [remapper.RangeThreshStateRemapper(0,-5,5),
                  remapper.RangeThreshStateRemapper(1,-5,5)]

# Costs
state_fun = state_fn.BallSetFn(np.zeros(2), 0.5)
#state_fun = state_fn.GaussianFn(1,np.zeros(2),9)
cost_function = costs.CostWrapper(state_fun)

# Actions:
actions = np.array([[-1],[0],[1]])

# Discount
discount = 0.997

# Samples
samples = 5

builder = mdp_builder.MDPBuilder(transition_function,
                                 discretizer,
                                 state_remappers,
                                 cost_function,
                                 actions,
                                 discount,
                                 samples,
                                 False)
mdp_obj = builder.build_mdp()


discrete_trans = mdp_wrapper.MDPTransitionWrapper(mdp_obj)
points = discretizer.get_cutpoints()
N = mdp_obj.num_states

start = np.random.randint(N)

I = 25
H = 100
for i in xrange(I):
    T = np.empty(H,dtype='i')
    s = start
    for h in xrange(H):
        s = discrete_trans.transition(np.full((1,1),s),0)
        T[h] = s
    plt.plot(points[T,0],
             points[T,1])
plt.show()

