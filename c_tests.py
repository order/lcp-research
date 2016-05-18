import cdiscrete as cd
import numpy as np
from discrete import make_points
import utils

import discrete

N = 5

l = np.array([0,0],dtype=np.double)
h = np.array([1,1],dtype=np.double)
n = np.array([N,N],dtype=np.uint64)

X = make_points([np.linspace(0,1,N+1)]*2)
c = 4*np.pi*np.array([0.5,0.5])
y = np.hstack([np.sin(X.dot(c)),0])

E = 50
eps = 0.1
P = make_points([np.linspace(-eps,1 + eps,E)]*2)

I = cd.interpolate(y,P,l,h,n)

utils.scatter_knn(I,P,3,100)
