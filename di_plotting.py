import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

from utils import *
from discrete import make_points

marsh = Marshaller()
setup = marsh.load('cdiscrete/test.mcts')


sim = marsh.load('cdiscrete/test.mcts.sim')
(ret,traj,dec,costs) = sim

(N,D,T) = traj.shape

xl = np.min(traj[:,0,:])
xh = np.max(traj[:,0,:])
vl = np.min(traj[:,1,:])
vh = np.max(traj[:,1,:])

knn = neighbors.KNeighborsRegressor(n_neighbors=1)
X = np.array([traj[:,0,0], traj[:,1,0]]).T
knn.fit(X,ret)

G = 150
(P,(X,Y)) = make_points([np.linspace(xl,xh,G),
                         np.linspace(vl,vh,G)],True)

Z = knn.predict(P)

plt.pcolormesh(X,Y,np.reshape(Z,(G,G)))

for i in xrange(N):
    plt.plot(traj[i,0,:],
             traj[i,1,:],
             'x-k')
plt.show()
