import mdp.double_integrator as di
import utils
import numpy as np
import matplotlib.pyplot as plt

params = utils.kwargify(step=0.01,
                       num_steps=1,
                       dampening=0.95,
                       control_jitter=0.1)
DI = di.DoubleIntegratorTransitionFunction(**params)

start = np.random.rand(1,2)
for i in xrange(25):
    curr = start
    for j in xrange(1000):
        new = DI.transition(curr,1)
        plt.plot([curr[0,0],new[0,0]],
                 [curr[0,1],new[0,1]],'-b')
        curr = new
plt.show()
