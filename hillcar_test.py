import numpy as np
from mdp.hillcar import basic_slope
import matplotlib.pyplot as plt

R = 15
N = 1000

x = np.linspace(-R,R,N)

dh = basic_slope(x,bowl=3,hill=3)
h = np.cumsum(dh) / N * (2*R)

plt.plot(x,h,x,dh,'--')
plt.show()

