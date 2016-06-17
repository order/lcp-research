import numpy as np
import scipy as sp
import scipy.fftpack as fft
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from discrete import make_points
from mdp import *

N = 128
D = 2
P = make_points([np.linspace(0,1,N+1)[:-1]]*D)
f = np.cos(2*np.pi*(1*P[:,0] + 1*P[:,1])).reshape(*[N]*D)
#f = np.random.rand(*[N]*D)

freq,shift,amp = top_trig_features(f,f.size-1,1e-12)

fn = TrigBasis(freq,shift)
B = fn.get_orth_basis(P)

Proj = B.dot(B.T)

rf = Proj.dot(f.flatten()).reshape(*[N]*D)

plt.subplot(2,2,1)
plt.imshow(f,interpolation='none')
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(rf,interpolation='none')
plt.colorbar()
plt.show()
