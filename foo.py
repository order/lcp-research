import numpy as np
import scipy.sparse as sps
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from utils.archiver import Unarchiver,read_shewchuk
import tri_mesh_viewer as tmv

unarch = Unarchiver("/home/epz/data/minop/sensitivity.sens")
(nodes,faces) = read_shewchuk("/home/epz/data/minop/sensitivity")

(N,_) = nodes.shape

p = unarch.p
twiddle = unarch.twiddle
jitter = unarch.jitter
noise = unarch.noise

value_diff = jitter - p[:,np.newaxis]
noise_diff = noise - np.mean(noise,0)[np.newaxis,:]

R = np.zeros(N)
for i in xrange(N):
    (r,p) = pearsonr(value_diff[i,:],noise_diff[i,:])
    if p < 0.005:
        R[i] = r 

plt.figure();
plt.subplot(1,2,1)
tmv.plot_vertices(nodes,faces,twiddle)
plt.subplot(1,2,2)
tmv.plot_vertices(nodes,faces,R)

plt.show()
