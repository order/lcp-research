from mdp.node_mapper import *
from mdp.state_remapper import *
from mdp.discretizer import *
from mdp.costs import *
from mdp.double_integrator import *

import matplotlib.pyplot as plt
import time

import lcp.solvers

import numpy as np

x_lim = 1
x_n = 50
xid = 0

v_lim = 3
v_n = 40
vid = 1

a_lim = 1
a_n = 5

MaxIter = 250
OOBCost = 25

cost_coef = np.array([2,1])

basic_mapper = InterpolatedGridNodeMapper(np.linspace(-x_lim,x_lim,x_n),np.linspace(-v_lim,v_lim,v_n))
assert(basic_mapper.get_num_nodes() == x_n * v_n)
physics = DoubleIntegratorRemapper()
cost_obj = QuadraticCost(cost_coef)
actions = np.linspace(-a_lim,a_lim,a_n)

left_oob_mapper = OOBSinkNodeMapper(xid,-float('inf'),-x_lim,basic_mapper.num_nodes)
right_oob_mapper = OOBSinkNodeMapper(xid,x_lim,float('inf'),basic_mapper.num_nodes+1)
state_remapper = RangeThreshStateRemapper(vid,-v_lim,v_lim)

# Add oob as exceptions to the cost
cost_obj.override[left_oob_mapper.sink_node] = OOBCost
cost_obj.override[right_oob_mapper.sink_node] = OOBCost

discretizer = ContinuousMDPDiscretizer(physics,basic_mapper,cost_obj,actions)
discretizer.add_state_remapper(state_remapper)
discretizer.add_node_mapper(left_oob_mapper)
discretizer.add_node_mapper(right_oob_mapper)

mdp_obj = discretizer.build_mdp()
print 'Built...', mdp_obj
lcp_obj = mdp_obj.tolcp()
print 'Built...', lcp_obj

vi = lcp.solvers.ValueIterator(mdp_obj)
solver = lcp.solvers.IterativeSolver(vi)
solver.termination_conditions.append(lcp.util.MaxIterTerminationCondition(MaxIter))
solver.solve()

Img = np.reshape(vi.v[:basic_mapper.get_num_nodes()],(x_n,v_n),order='F')
#Img = np.reshape(mdp_obj.costs[1][:basic_mapper.get_num_nodes()],(v_n,x_n))
plt.imshow(Img)
plt.show()
