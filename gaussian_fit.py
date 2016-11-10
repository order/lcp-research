import numpy as np
import matplotlib.pyplot as plt
from utils import make_points

def gauss(X,a,C):
    (N,D) = X.shape
    g = np.empty(N)
    for i in xrange(N):
        x = X[i,:]
        g[i] = a * np.exp(-0.5 * x.dot(C.dot(x.T)))
    return g

def gradient(Y,X,a,C):
    (N,D) = X.shape
    
    F = gauss(X,a,C)
    da = -np.sum((Y - F)*F)

    dC = np.zeros((D,D))
    for i in xrange(N):
        x = X[i,:]
        dC -= (Y[i] - F[i]) * F[i] * x.T * x
    
    return (da,dC)

def gradient_descent(X,Y):
    C = np.eye(2);
    a = 0.1
    for i in xrange(25):
        print 'a:',a
        print "\tResidual:", np.linalg.norm(Y - gauss(X,a,C))**2
        (da,dC) = gradient(Y,X,a,C)
        print "\tda:", da
        print "\tdC:", da

        a -= 1e-4 * da
        C -= 1e-4 * dC
    return (a,C)
    

G = 127
(P,[X,Y]) = make_points([np.linspace(-1,1,G)]*2,True)

Z = gauss(P,2,np.eye(2))
Z[np.where(np.isnan(Z))[0]] = 1
(a,C) = gradient_descent(P,Z)

plt.subplot(2,2,1)
plt.pcolormesh(X,Y,np.reshape(Z,X.shape))
plt.subplot(2,2,2)
plt.pcolormesh(X,Y,np.reshape(gauss(P,a,C),X.shape))
plt.subplot(2,2,3)
plt.pcolormesh(X,Y,np.reshape(Z - gauss(P,a,C),X.shape))
plt.colorbar()
plt.show()
