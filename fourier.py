import numpy as np
import scipy as sp
import scipy.fftpack as fft
import scipy.sparse as sps
from mpl_toolkits.mplot3d import Axes3D


from discrete import make_points
from mdp import *

import matplotlib.pyplot as plt

N = 8
M = 4
x = np.linspace(0,1,(N+1))
X = np.linspace(0,1,(N*M+1))

plt.plot(x,np.sin(2*np.pi*x))
plt.plot(X,np.sin(2*np.pi*X))
plt.plot(X,np.sin(7*np.pi*X))
plt.show()
