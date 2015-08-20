from mdp.node_mapper import *
import time

import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 10
D = 2
M = 3
P = np.random.rand(N,D)
#Nodes = np.random.randint(M**2,size=(N,1))
Nodes = np.array(range(M**2))

grid = InterpolatedGridNodeMapper(0,np.linspace(0,1,M),np.linspace(0,1,M))
States = grid.nodes_to_states(Nodes)
print States
#NodeDists = grid.states_to_node_dists(States)

#for i in xrange(len(Nodes)):
#   print Nodes[i], NodeDists[i]