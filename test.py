import mdp.double_integrator as di
import utils
import numpy as np
import matplotlib.pyplot as plt

params = utils.kwargify(step=0.01,
                        num_steps=1,
                        dampening=0.01,
                        control_jitter=0.5)
DI = di.DoubleIntegratorTransitionFunction(**params)

start = np.random.rand(1,2)
plt.plot(start[0,0],start[0,1],'rp')

R = 25
H = 2500
for i in xrange(R):
    curr = start
    Runs = np.empty((H,2))
    for j in xrange(H):
        new = DI.transition(curr,-curr[0,0]-0.1*curr[0,1])[0,:,:]
        curr = new
        Runs[j,:] = curr.flatten()

    plt.plot(Runs[:,0],Runs[:,1],'-b',alpha=0.25)
plt.show()
