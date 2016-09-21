import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import make_points

G = 160
grids = [np.linspace(-1,1,G)]*2
[P,[X,Y]] = make_points(grids,True)

sqnorm = lambda x: np.sum(x**2,1)
norm = lambda x: np.sqrt(sqnorm(x))

b = 5
f = np.cos(b*b*norm(P)) * np.exp(-b*sqnorm(P))

Z = np.reshape(f,(G,G))

cmap = plt.get_cmap("jet")

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z,
                rstride=1,
                cstride=1,
                cmap=cmap,lw=0)
plt.show()
