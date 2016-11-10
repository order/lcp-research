import numpy as np
import matplotlib.pyplot as plt
from utils import make_points

def gauss(X,a,C):
    (N,D) = X.shape
    g = np.empty(N)
    for i in xrange(N):
        x = X[i,:]
        g[i] = a * np.exp(-x.dot(C.dot(x.T)))
    return g

def gradient(Y,X,a,C):
    (N,D) = X.shape
    
    F = gauss(X,a,C)
    da = -np.sum((Y - F)*F)

    dC = np.zeros((D,D))
    for i in xrange(N):
        x = X[i,:]
        dC += (Y[i] - F[i]) * F[i] * np.outer(x,x)
    assert np.linalg.norm(dC - dC.T) < 1e-15
        
    return (da,dC)

def gradient_descent(X,Y):

    N = Y.size
    
    f = lambda a,C: np.linalg.norm(Y - gauss(X,a,C))**2
    
    C = np.random.rand(2,2)
    C = C.T.dot(C)
    assert np.all(np.linalg.eigvals(C) > 0)
    a = 2 + (2*np.random.rand()-1)
    for i in xrange(250):
        print 'a:',a
        F = f(a,C)
        print "\tAverage Residual:", F / float(N)

        if F / float(N) < 1e-4:
            break
        
        (da,dC) = gradient(Y,X,a,C)
        print "\tda:", da
        print "\tdC:", dC
        
        t = 1.0 / np.linalg.norm(dC)
        t0 = 1e-9
        s = 0.1
        b = 0.5
        grad_norm = da**2 + np.sum(dC**2)
        
        while True:
            try:
                np.linalg.cholesky(C - t * dC)
            except Exception:
                print "\tNon-PD, t=",t
                t *= b
                continue
            break
        while np.any(np.linalg.eigvals(C - t * dC) < 0):
            t *= b

        assert np.all(np.linalg.eigvals(C - t * dC) > 0)
        
        while f(a - t *da, C - t * dC) - F > - s * t * grad_norm and t > t0:
            print "\tBacktracking",t,f(a + t *da, C + t * dC) - F
            t *= b
        assert np.all(np.linalg.eigvals(C - t * dC) > 0)

        a -= t * da
        C -= t * dC
    return (a,C)
    

G = 128
(P,[X,Y]) = make_points([np.linspace(-5,5,G)]*2,True)

Z = gauss(P,2,np.array([[3,-1],[-1,2]]))
Z[np.where(np.isnan(Z))[0]] = 1
(a,C) = gradient_descent(P,Z)
print "Final a:",a
print "Final C:",C

plt.subplot(2,2,1)
plt.pcolormesh(X,Y,np.reshape(Z,X.shape))
plt.subplot(2,2,2)
plt.pcolormesh(X,Y,np.reshape(gauss(P,a,C),X.shape))
plt.subplot(2,2,3)
plt.pcolormesh(X,Y,np.reshape(Z - gauss(P,a,C),X.shape))
plt.colorbar()
plt.show()
