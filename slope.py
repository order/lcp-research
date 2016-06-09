import numpy as np
from mdp.transitions import new_slope

import matplotlib.pyplot as plt

A = -2
B = 6
G = 251
x = np.linspace(A,B,G)
S = new_slope(x)
H = np.cumsum(S) * (B - A) / G


l = min(np.min(S),np.min(H))*1.1
u = max(np.max(S),np.max(H))*1.1
plt.plot(x,S,x,H,[0,0],[l,u],[4,4],[l,u])
plt.show()


