import numpy as np
from utils.archiver import Unarchiver
import matplotlib.pyplot as plt

unarch = Unarchiver('cdiscrete/test.sol')

N = unarch.p.size / 4
exact_p = np.reshape(unarch.p,(N,4),order='F');
smooth_p = np.reshape(unarch.sp,(N,4),order='F')
proj_p = np.reshape(unarch.pp,(N,4),order='F')
smooth_proj_p = np.reshape(unarch.spp,(N,4),order='F')

P = [exact_p,smooth_p,proj_p,smooth_proj_p]
names = ['Exact','Smooth','Proj', 'Smooth proj']
plt.figure()

for p in P:
    plt.plot(p[:,0])
plt.title('Value')
plt.legend(names, loc='best')

plt.figure()
plt.suptitle('Flow')

for (i,(p,name)) in enumerate(zip(P,names)):
    plt.subplot(2,2,i+1)
    plt.title(name)
    for i in xrange(1,4):
        plt.semilogy(p[:,i])

plt.show()
