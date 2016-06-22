import numpy as np
import matplotlib.pyplot as plt
from utils.marshal import *

marsh = Marshaller()
(gain,traj,decisions,costs) = marsh.load('cdiscrete/test.mcts.sim')

(N,D,T) = traj.shape

print traj.T

quit()

B = np.max(np.abs(traj[:]))*1.1
(X,Y) = np.meshgrid(*[np.linspace(-B,B,33)]*2)
Z = (np.sqrt(X*X + Y*Y) < 0.25)
plt.pcolormesh(X,Y,Z,alpha=0.25,lw=0)

for i in xrange(N):
    plt.plot(traj[i,0,:],traj[i,1,:],'.-k',alpha=0.25)

plt.show()

######################
# TODO:
# Run with super long rollout (check that it works with just simulation)
# Run with more accurate Q
# Do coordinate descent w/ restarts to find good parameters once its working.
