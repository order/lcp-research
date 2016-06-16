import numpy as np
import scipy as sp
import scipy.fftpack as fft
import scipy.sparse as sps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from discrete import make_points
from mdp import *

def find_ind_col(A):
    (N,D) = A.shape
    idx = []
    nidx = []
    for i in xrange(D):
        if min(N,D) == len(idx):
            nidx.extend(xrange(i,D))
            break
        idx.append(i)
        if np.linalg.matrix_rank(A[:,idx]) == len(idx):
            continue
        idx.pop()
        nidx.append(i)
    return (idx, nidx)

N = 2
D = 3

F = make_points([np.arange(N,dtype=np.double)]*D)
P = F / N
F *= 2*np.pi

C = P.dot(F.T)

A = np.hstack([np.cos(C),np.sin(C)])
(idx,nidx) = find_ind_col(A)

A[:,nidx] = 0
#plt.imshow(np.cos(C).dot(np.cos(C).T),interpolation='none')
plt.imshow(A,interpolation='none')
plt.show()

quit()




def full_basis(lens):
    F = make_points([np.arange(N) for N in lens])
    n = F.shape[0]
    
    return (F,np.zeros(n))
    
    #B = np.vstack([np.flipud(F),F])
    #s = np.concatenate([np.zeros(n),
    #                    np.pi*0.5*np.ones(n)])
    #return (B,s)

N = 4
M = 64
D = 2

(p,(x,y)) = make_points([np.linspace(0,1,(N+1))[:-1]]*D,True)
(P,(X,Y)) = make_points([np.linspace(0,1,(M+1))[:-1]]*D,True)

(f,s)  = full_basis([N]*D)
print 'Shape from basis:',f.shape
#print np.hstack([f,s[:,np.newaxis]])

scale = 2.0*np.pi
basis = TrigBasis(scale*f,s)
b = basis.get_basis(p)
B = basis.get_basis(P)

#b = sp.linalg.hadamard(p.shape[0])
#B = sp.linalg.hadamard(P.shape[0])

ax = plt.subplot(1,3,1,projection='3d')
k = 2
ax.plot_surface(X,Y,B[:,k].reshape(M,M),rstride=1,
                cstride=1,cmap='plasma',alpha=0.25,lw=0)
ax.plot_wireframe(X,Y,B[:,k].reshape(M,M),edgecolor='k',rstride=M/N,
                  cstride=M/N,alpha=0.25)
ax.scatter(p[:,0],p[:,1],b[:,k])
ax.set_xlabel('x')
ax.set_ylabel('y')

ax = plt.subplot(1,3,2)
plt.imshow(b,interpolation='none')
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(b.T.dot(b),interpolation='none')
plt.colorbar()
print np.median(b.T.dot(b))
plt.show()
