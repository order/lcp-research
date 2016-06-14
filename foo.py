import numpy as np
import scipy as sp
import scipy.fftpack as fft
import scipy.sparse as sps
from mpl_toolkits.mplot3d import Axes3D


from discrete import make_points
from mdp import *

import matplotlib.pyplot as plt

# Build the model
N = 16
D = 3
P = make_points([np.linspace(0,1,N+1)[:-1]]*D)

f = np.empty(N**D)
for _ in xrange(12):
    freq = np.random.randint(N,size=(D,))
    shift = np.random.randint(2)
    f += np.sin(2*np.pi * P.dot(freq) + np.pi / 2.0 * shift) \
         * np.exp(-np.linalg.norm(freq)/10)
#f = np.sin(2*np.pi*P[:,1])
f = f.reshape(*[N]*D)

freq,shift,amp = top_trig_features(f,N**D)

print freq.shape

# Eval
approx_f = TrigFunction(freq,shift,amp[:,np.newaxis])
r = approx_f.evaluate(P).reshape(*[N]*D)

print 'Error: ', np.linalg.norm(r - f) / np.linalg.norm(f)

if D == 2:

    plt.subplot(2,2,1)
    plt.imshow(f,interpolation='none')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(r,interpolation='none')
    plt.colorbar()

    F = np.fft.fftn(f)
    plt.subplot(2,2,3)
    plt.imshow(np.abs(F),interpolation='none')
    plt.colorbar()

    R = np.fft.fftn(r)
    plt.subplot(2,2,4)
    plt.imshow(np.abs(R),interpolation='none')
    plt.colorbar()

    plt.show()
   
