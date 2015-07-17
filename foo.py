import numpy as np
import matplotlib.pyplot as plt

def fb(x):
    return np.linalg.norm(np.sqrt(x**2) - x,axis=1)

N = 2500
X = 2*np.random.rand(N,2)-1
Y = 2*np.random.rand(N,2)-1
t = np.random.rand(N)
T = np.column_stack((t,t))

outer = t*fb(X) + (1-t)*fb(Y)
inner = fb(T*X + (1-T)*Y)

plt.scatter(outer,inner)
plt.show()
