import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


from regular_interpolate import RegularGridInterpolator
Grid = [(0,1,1),(0,1,1)]
interp = RegularGridInterpolator(Grid)
cpts = interp.get_cutpoints()
plt.plot(cpts[:,0],cpts[:,1],'.k')

point = np.array([[-1, 2]])
plt.plot(point[:,0],point[:,1],'go')

dist = interp.points_to_index_distributions(point)
#assert((dist.sum() - M) / float(M) < 1e-15)


plt.subplot(1,2,1)
plt.spy(dist)

plt.subplot(1,2,2)
plt.plot(point[:,0],point[:,1],'go')
N = dist.nnz
for i in xrange(N):
    c = dist.col[i]
    r = dist.row[i]
    plt.plot([point[c,0],cpts[r,0]],
             [point[c,1],cpts[r,1]],'-b')

recon = dist.T.dot(cpts)
print recon


plt.plot(recon[:,0],recon[:,1],'rs')
plt.show()
 
