import numpy as np
import scipy.sparse as sps
import scipy.signal as signal

from lcp import *
from config.solver import *

import matplotlib.pyplot as plt

def rect(x):
    return np.maximum(0,x)

def r_res(x,M,q):
    return np.linalg.norm(np.minimum(x,M.dot(x)+q))

def s_res(x,M,q):
    w = M.dot(x) + q
    res = np.hstack([rect(-x),
                     rect(-w),
                     x.dot(w)])
    return np.linalg.norm(res)

def get_smoothed_random(N,w=25):
    assert(w > 2)
    x = np.random.rand(N+w-1)
    x *= float(N) / np.sum(x)
    win = signal.hann(w)
    smooth = signal.convolve(x,win,mode='valid')/np.sum(win)
    assert((N,) == smooth.shape)
    return smooth

def build_lcp(a,b):
    N = a.size
    p = -np.ones(N)/float(N)
    q = np.hstack([p,a,b])
    assert((3*N,) == q.shape)
    
    I = sps.eye(N)
    M = sps.bmat([[None,I,I],[-I,None,None],[-I,None,None]])
    assert((3*N,3*N) == M.shape)

    return LCPObj(M,q)

def build_chebyshev_basis(N,k):
    x = np.linspace(-1,1,N)

    B = np.polynomial.chebyshev.chebvander(x,k)
    B = orthonorm(B)
    P = sps.block_diag([B]*3)
    return P

def build_plcp(P,lcp):
    U = P.T.dot(lcp.M)
    q = P.dot(P.T.dot(lcp.q))
    
    return ProjectiveLCPObj(P,U,U,q)

def kojima_solve(lcp,**kwargs):
    # Solve
    (N,) = lcp.q.shape
    x0 = kwargs.get('x0',np.ones(N))
    y0 = kwargs.get('y0',np.ones(N))
    start = time.time()
    (p,d,data) = solve_with_kojima(lcp,
                                   thresh=1e-6,
                                   max_iter=150,
                                   x0=x0,
                                   y0=y0)
    print 'Kojima ran for:', time.time() - start, 's'
    return (p,d,data)

def projective_solve(plcp):
    (N,K) = plcp.Phi.shape
    x0 = np.ones(N)
    y0 = np.ones(N)
    w0 = np.zeros(K) 
    start = time.time()
    (p,d,data) = solve_with_projective(plcp,
                                       thresh=1e-12,
                                       max_iter=250,
                                       x0=x0,
                                       y0=y0,
                                       w0=w0)
    print 'Projective ran for:', time.time() - start, 's'
    return (p,d,data)

if __name__ == "__main__":

    N = 256 # Vector length
    w = 16 # Window size (smoothing)
    T = 1 # Trials

    num_basis = range(16,128,2) # Basis size to try
    K = len(num_basis)
    
    if True:
        LCPError = np.empty((T,K))
        LCPSol = np.empty((T,K,N))
        ProjError = np.empty((T,K))
        ProjSol = np.empty((T,K,N))
        R = np.empty((T,K))
        S = np.empty((T,K))
        BasisRes = np.empty((T,K))
                        
        for t in xrange(T):
            a = get_smoothed_random(N,w)
            b = get_smoothed_random(N,w)
            ab = np.minimum(a,b)            
            lcp = build_lcp(a,b)
            M = lcp.M
            q = lcp.q
            for (i,k) in enumerate(num_basis):
                P = build_chebyshev_basis(N,k)

                plcp = build_plcp(P,lcp)
                (p,d,data) = projective_solve(plcp)
                lcp_error = np.linalg.norm(ab - p[:N])

                B = (P.tolil())[:N,:N]
                proj_p = B.dot(B.T.dot(ab))
                proj_err = np.linalg.norm(ab - proj_p)

                LCPError[t,i] = lcp_error
                LCPSol[t,i,:] = p[:N]
                ProjError[t,i] = proj_err
                ProjSol[t,i,:] = proj_p[:N]
                R[t,i] = r_res(p,M,q)
                S[t,i] = s_res(p,M,q)
                BasisRes[t,i] = np.linalg.norm(p - P.dot(P.T.dot(p)))

    t = 0
    plt.figure()
    plt.semilogy(num_basis,LCPError[t,:])
    plt.semilogy(num_basis,ProjError[t,:])
    plt.semilogy(num_basis,R[t,:])
    plt.semilogy(num_basis,S[t,:])
    plt.semilogy(num_basis,BasisRes[t,:])
    plt.legend(['LCP error',
                'Projection error',
                'R residual',
                'S residual',
                'Basis residual'])


    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(LCPSol[t,...] - ab[np.newaxis,:],'-b',alpha=0.2)
    plt.title('LCP residual')
    plt.subplot(1,2,2)
    plt.plot(ProjSol[t,...] - ab[np.newaxis,:],'-r',alpha=0.2)
    plt.title('Projection residual')
    
    plt.show()
    
            
