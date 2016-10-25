import numpy as np
import matplotlib.pyplot as plt

data = np.load("test.npz")
mu = data['mu']
d = data['dims']


plt.figure()
plt.plot(d,mu,'.-')
plt.xlabel('size')
plt.ylabel('mu')
plt.title('l1-norm mu')

plt.figure()
plt.plot(np.diff(mu),'o-')
plt.ylabel('mu difference')
plt.title('l1-norm mu difference')

plt.show()
