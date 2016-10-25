import numpy as np
from utils.archiver import Unarchiver
import matplotlib.pyplot as plt

unarch = Unarchiver('cdiscrete/test.sol')

plt.figure()
plt.imshow(unarch.M.toarray(),interpolation='nearest')
plt.colorbar()

p = unarch.p
N = p.size / 4
P = np.reshape(unarch.p,(N,4),order='F');

plt.figure()
plt.plot(P[:,0])

plt.show()
