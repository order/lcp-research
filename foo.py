import numpy as np
import matplotlib.pyplot as plt
from utils.archiver import *
from scipy.stats import pearsonr

import tri_mesh_viewer as tmv

(nodes,faces) = read_shewchuk("/home/epz/data/minop/sensitivity")
unarch = Unarchiver("/home/epz/data/minop/sensitivity.exp_res")

(N,D) = nodes.shape

assert(N == unarch.twiddle.size)

plt.subplot(1,2,1)
tmv.plot_vertices(nodes,faces,unarch.twiddle)

J = unarch.jitter
R = unarch.noise

P = np.zeros(N)
for i in xrange(N):
    (r,p) = pearsonr(J[i,:],R[i,:])
    if(p < 0.01):
        P[i] = r

plt.subplot(1,2,2)
tmv.plot_vertices(nodes,faces,P)
plt.show()
