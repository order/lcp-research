import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from irregular_interpolate import IrregularGridInterpolator

M = 100
X = 10
K = 15
x_grid = np.sort(X*np.random.rand(K))
y_grid = np.sort(X*np.random.rand(K))
Grid = [x_grid,y_grid]

interp = IrregularGridInterpolator(Grid)
cpts = interp.get_cutpoints()
plt.plot(cpts[:,0],cpts[:,1],'.k')

point = (X + 2)*np.random.rand(M,2) - 1
#point = -1*np.ones((1,2))
x_oob = np.logical_or(point[:,0] < x_grid[0],
                      point[:,0] > x_grid[-1])
y_oob = np.logical_or(point[:,1] < y_grid[0],
                      point[:,1] > y_grid[-1])
oob = np.logical_or(x_oob,y_oob)
plt.plot(point[:,0],point[:,1],'go')

dist = interp.points_to_index_distribution(point)
#assert((dist.sum() - M) / float(M) < 1e-15)

N = dist.nnz
for i in xrange(N):
    c = dist.col[i]
    r = dist.row[i]
    plt.plot([point[c,0],cpts[r,0]],
             [point[c,1],cpts[r,1]],'-b')

dist = dist.toarray()
recon = np.dot(dist.T,cpts)
assert(not np.any(np.isnan(recon[~oob,:])))

plt.plot(recon[~oob,0],recon[~oob,1],'rs')
plt.show()
assert(np.linalg.norm(recon[~oob,:] - point[~oob,:]) < 1e-12)
 
