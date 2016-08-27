import numpy as np
import scipy.spatial as spt
from utils import make_points

import matplotlib.pyplot as plt

N = 5
points = make_points([np.linspace(0,1,5)]*2)
points = np.vstack([points,np.random.rand(25,2)])
tri = spt.Delaunay(points,False,True)

target = np.random.rand(2)
simplex = tri.find_simplex(target)
vertices = tri.simplices[simplex,:]
print 'Target:',target
print 'Simplex:',simplex
print 'Vertices:', vertices


plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], '.b')
plt.fill(points[vertices,0],points[vertices,1],'r',alpha=0.25)
plt.plot(target[0],target[1],'y*',lw=2,markersize=15)
plt.show()
