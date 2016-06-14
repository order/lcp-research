import numpy as np
import scipy as sp
import scipy.fftpack as fft
import scipy.sparse as sps

import matplotlib.pyplot as plt

N = 256
sigma = 1
x = np.linspace(-10,10,N)
g = np.exp(-x*x / sigma)
g = g / np.sum(g)

h = np.convolve(np.random.randn(N),g,mode='same')

f = fft.fft(h)

r = np.zeros(N)
for i in xrange(N/2+1):
    if i == 0 or i == N/2:
        Re = np.real(f[i])/N
    else:
        Re = 2*np.real(f[i])/N
    Im = -2 * np.imag(f[i]) / N
    
    r += Re*np.cos(2*np.pi*i*np.arange(N)/N)
    r += Im*np.sin(2*np.pi*i*np.arange(N)/N)
plt.plot(x,h,x,r)
plt.show()
