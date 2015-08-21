from mdp.node_mapper import *
from mdp.state_remapper import *
from mdp.discretizer import *
from mdp.costs import *
from mdp.double_integrator import *
import time

import numpy as np

x_lim = 1
x_n = 25
xid = 0

v_lim = 3
v_n = 15
vid = 1

a_lim = 1
a_n = 5

cost_coef = np.array([2,1])

basic_mapper = InterpolatedGridNodeMapper(np.linspace(-x_lim,x_lim,x_n),np.linspace(-v_lim,v_lim,v_n))
physics = DoubleIntegratorRemapper()
cost_obj = QuadraticCost(cost_coef)
actions = np.linspace(-a_lim,a_lim,a_n)

left_oob_mapper = OOBSinkNodeMapper(xid,-float('inf'),-x_lim,basic_mapper.num_nodes)
right_oob_mapper = OOBSinkNodeMapper(xid,x_lim,float('inf'),basic_mapper.num_nodes+1)
state_remapper = RangeThreshStateRemapper(vid,-v_lim,v_lim)

discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
discretizer.add_state_remapper(state_remapper)
discretizer.add_node_mapper(left_oob_mapper)
discretizer.add_node_mapper(right_oob_mapper)

MDP = discretizer.build_mdp()