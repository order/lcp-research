import numpy as np
import scipy as sp
import scipy.sparse as sps

from discretize import make_points

from mdp import InterpolatedFunction

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


from regular_interpolate import RegularGridInterpolator
Grid = [(0,1,1),(0,1,1)]
interp = RegularGridInterpolator(Grid)
cpts = interp.get_cutpoints()

point = np.array([[2, -1]])

dist = interp.points_to_index_distributions(point)

#assert((dist.sum() - M) / float(M) < 1e-15)

if False:
    plt.subplot(1,2,1)
    plt.spy(dist)

    plt.subplot(1,2,2)
    plt.plot(point[:,0],point[:,1],'go')
    N = dist.nnz
    for i in xrange(N):
        c = dist.col[i]
        r = dist.row[i]
        if interp.is_oob(r):
            continue
        plt.plot([point[c,0],cpts[r,0]],
                 [point[c,1],cpts[r,1]],'-b')

    plt.show()


G = 25
(P,(X,Y)) = make_points([np.linspace(-0.2,1.2,G)]*2,True)
print P
print interp.to_cell_coord(P)
print interp.to_indices(P)
print interp.points_to_index_distributions(P)
v = np.arange(interp.num_nodes())
fn = InterpolatedFunction(interp,v)
Z = fn.evaluate(P)
Img = np.reshape(Z,(G,G))

if True:
    plt.pcolor(X,Y,Img)
    plt.show()
