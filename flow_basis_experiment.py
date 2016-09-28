import numpy as np
import matplotlib.pyplot as plt

from utils import make_points
from utils.archiver import Unarchiver
import sys

(_,filename) = sys.argv

unarch = Unarchiver(filename)
assert hasattr(unarch,"res_data")
data = unarch.res_data

(N,R) = data.shape

idx = unarch.num_basis
IDX = np.tile(np.reshape(idx,(N,1)),(1,R))

plt.plot(IDX.ravel(),data.ravel(),'.')

smoother = [0.05,0.2,0.5,0.2,0.05]
assert(idx.size >=5)

p25 = np.convolve(np.percentile(data,25,1),smoother,'valid')
p50 = np.convolve(np.median(data,1),smoother,'valid')
p75 = np.convolve(np.percentile(data,75,1),smoother,'valid')
plt.plot(idx[2:-2],p25,'r--',lw=2)
plt.plot(idx[2:-2],p50,'r-',lw=2)
plt.plot(idx[2:-2],p75,'r--',lw=2)
#plt.ylim([np.min(data),np.max(data)])
plt.show()
