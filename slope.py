import numpy as np
import math
import matplotlib.pyplot as plt

G = 9.806

def slope(x,points):
    t_crit = -math.asin(1.0/G)
    s = -0.05*t_crit
    eps = -0.01*t_crit

    y = np.zeros(x.shape)

    mask = (x < points[0])
    y[mask] = (t_crit + s) * x[mask] / points[0]

    mask = np.logical_and(points[0]<= x, x < points[1])
    y[mask] = (t_crit + s)
    
    mask = np.logical_and(points[1]<= x, x < points[2])
    y[mask] = (-2*t_crit - s + eps) * (x[mask] - points[1]) / (points[2] - points[1]) + t_crit + s

    mask = (points[2] < x)
    y[mask] = -t_crit + eps    

    return y

x = np.linspace(-2,5,350)

t = slope(x,(1,2,3))
dh = np.tan(t)

plt.plot(x,t,x,dh,x,np.cumsum(dh)/50)
plt.show()
        
