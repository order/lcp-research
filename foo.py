from mdp.node_mapper import *

import numpy as np
import scipy.interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N = 5
D = 2
P = np.random.rand(N,D)
#print P

low = 0
high = 1

#grid = PiecewiseConstRegularGridNodeMapper(2,(low,high,3),(low,high,4),(low,high,2))
#grid = PiecewiseConstRegularGridNodeMapper(2,(0,1,4),(0,1,2))
grid = InterpolatedGridNodeMapper(0,np.linspace(0,1,4),np.linspace(0,1,3))

if False:
    GridNodes = grid.states_to_node_dists(P,set())
    #for (i,nd) in GridNodes.items():
        #print i,nd
    Pn = grid.nodes_to_states(nodemap_to_nodearray(GridNodes))
    #print Pn
    f = plt.figure()
    ax = f.add_subplot(111,projection='3d')
    #ax = f.add_subplot(111)

    for i in xrange(P.shape[0]):
        ax.plot([P[i,0],Pn[i,0]],[P[i,1],Pn[i,1]],[P[i,2],Pn[i,2]],'b-',alpha=0.2)
        
    plt.show()

if True:
    GridNodes = grid.states_to_node_dists(P,set())
    for (i,nd) in GridNodes.items():
        print '[{0}] {1}'.format(i,nd)
    grid.build_node_id_cache()
    GridNodes = grid.states_to_node_dists(P,set())
    print '-'*10
    for (i,nd) in GridNodes.items():
        print '[{0}] {1}'.format(i,nd)    
        
        
if False:
    GridNodes = grid.states_to_node_dists(P,set())
    Colors = np.array([min(nd.keys()) for nd in GridNodes.values()])
    #plt.scatter(X[:,0],X[:,1],s=50,c=Colors)
    Xg,Yg = np.meshgrid(np.linspace(0,1,250),np.linspace(0,1,250))
    Cg = scipy.interpolate.griddata(P,Colors,(Xg,Yg),method='linear')
    plt.imshow(Cg)
    plt.show()