import numpy as np
from mdp import *

import matplotlib.pyplot as plt

boundary = HillcarBoundary([(-1,1),(-1,1)])

N = 50
P = np.random.randn(N,2)

Q = boundary.enforce(P)

for i in xrange(N):
    plt.plot([P[i,0],Q[i,0]],
             [P[i,1],Q[i,1]],'-o')
plt.show()

