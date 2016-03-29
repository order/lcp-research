import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from irregular_interpolate import IrregularGridInterpolator

M = 50
X = Y = 5
Grid = [(0,X,X),(0,Y,Y)]

interp = IregularGridInterpolator(Grid)
cpts = interp.get_cutpoints()
plt.plot(cpts[:,0],cpts[:,1],'.k')

point = (max(X,Y) + 2)*np.random.rand(M,2) - 1
plt.plot(point[:,0],point[:,1],'go')

dist = interp.points_to_index_distribution(point)
#assert((dist.sum() - M) / float(M) < 1e-15)
print dist

N = dist.nnz
for i in xrange(N):
    c = dist.col[i]
    r = dist.row[i]
    plt.plot([point[c,0],cpts[r,0]],
             [point[c,1],cpts[r,1]],'-b')

dist = dist.toarray()
recon = np.dot(dist.T,cpts)
print recon
plt.plot(recon[:,0],recon[:,1],'rs')
plt.show()
assert(np.linalg.norm(recon - point.squeeze()) < 1e-12)
 
