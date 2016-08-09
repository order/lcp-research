import numpy as np
import scipy.sparse as sps
import scipy.signal as signal

from lcp import *
from config.solver import *

import matplotlib.pyplot as plt

def rect(x):
    return np.maximum(0,x)

def r_res(x,M,q,ord=2):
    return np.linalg.norm(np.minimum(x,M.dot(x)+q),ord)

def s_res(x,M,q,ord=2):
    w = M.dot(x) + q
    res = np.hstack([rect(-x),
                     rect(-w),
                     x.dot(w)])
    return np.linalg.norm(res,ord)

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

def build_chebyshev_basis(a,b,N,k):
    x = np.linspace(-1,1,N)

    B = np.polynomial.chebyshev.chebvander(x,k-3)
    B = np.hstack([a[:,np.newaxis],
                   b[:,np.newaxis],
                   B])
    B = orthonorm(B)
    P = sps.block_diag([B]*3)
    assert (3*N,3*k) == P.shape
    return P

def build_aug_plcp(P,lcp):
    U = P.T.dot(lcp.M)
    q = P.dot(P.T.dot(lcp.q))
    
    plcp = ProjectiveLCPObj(P,U,U,q)
    (aplcp,x0,y0,w0) = augment_plcp(plcp,10)
    return (plcp,aplcp,x0,y0,w0)

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

def projective_solve(plcp,x0,y0,w0):
    (N,K) = plcp.Phi.shape
    start = time.time()
    (p,d,data) = solve_with_projective(plcp,
                                       thresh=1e-22,
                                       max_iter=250,
                                       x0=x0,
                                       y0=y0,
                                       w0=w0)
    print 'Projective ran for:', time.time() - start, 's'
    return (p,d,data)

if __name__ == "__main__":

    N = 256 # Vector length
    window_size = 16 # Window size (smoothing)
    T = 1 # Trials

    num_basis = range(16,64,4) # Basis size to try
    K = len(num_basis)
    
    if True:
        LCPError = np.empty((T,K))
        LCPSol = np.empty((T,K,N))
        ExactDualRes = np.empty((T,K)) # d - (Mx+q)
        ApproxDualRes = np.empty((T,K)) # d - (PMP + (I-P))x - q
        
        IP = np.empty((T,K)) # <x,Mx+q>
        DualInfeas = np.empty((T,K)) # (-Mx-q)_+
        
        ProjError = np.empty((T,K))
        ProjSol = np.empty((T,K,N))
        R = np.empty((T,K))
        S = np.empty((T,K))
        BasisRes = np.empty((T,K))
                        
        for t in xrange(T):
            a = get_smoothed_random(N,window_size)
            b = get_smoothed_random(N,window_size)
            ab = np.minimum(a,b)            
            lcp = build_lcp(a,b)
            M = lcp.M
            q = lcp.q
            for (i,k) in enumerate(num_basis):
                # Build basis; make sure orthogonal
                P = build_chebyshev_basis(a,b,N,k)
                PtP = P.T.dot(P)
                Pi = P.dot(P.T)
                PiNull = (np.eye(3*N) - Pi)
                I = np.eye(3*k)
                assert np.linalg.norm(PtP - I) < 1e-12

                # Solve LCP
                (plcp,aplcp,x0,y0,w0) = build_aug_plcp(P,lcp)
                (aug_x,aug_y,data) = projective_solve(aplcp,x0,y0,w0)
                print 'Augmented pair:',(aug_x[-1],aug_y[-1])
                x = aug_x[:-1]
                y = aug_y[:-1]
                lcp_error = np.linalg.norm(ab - x[:N])

                # Project exact solution
                B = (P.tolil())[:N,:N]
                proj_ab = B.dot(B.T.dot(ab))
                proj_err = np.linalg.norm(ab - proj_ab)

                # Store data
                LCPError[t,i] = lcp_error
                LCPSol[t,i,:] = x[:N]

                w = lcp.F(x) # Exact dual from x
                u = plcp.F(x) # Projective dual from x
                
                ExactDualRes[t,i] = np.linalg.norm(y - w)
                ApproxDualRes[t,i] = np.linalg.norm(y - u)
                IP[t,i] = x.dot(w)
                DualInfeas[t,i] = np.linalg.norm(rect(-w))
                
                ProjError[t,i] = proj_err
                ProjSol[t,i,:] = proj_ab[:N]
                R[t,i] = r_res(x,M,q,np.inf)
                S[t,i] = s_res(x,M,q,np.inf)
                BasisRes[t,i] = np.linalg.norm(PiNull.dot(x))

    t = 0
    plt.figure()
    plt.semilogy(num_basis,LCPError[t,:])
    plt.semilogy(num_basis,ProjError[t,:])
    plt.semilogy(num_basis,R[t,:])
    plt.semilogy(num_basis,S[t,:])
    plt.semilogy(num_basis,S[t,:] + np.sqrt(S[t,:]))
    
    plt.legend(['LCP error',
                'Projection error',
                'R residual',
                'S residual',
                'S + S**0.5'])

    plt.figure()
    plt.semilogy(num_basis,ExactDualRes[t,:])
    plt.semilogy(num_basis,ApproxDualRes[t,:])
    plt.semilogy(num_basis,BasisRes[t,:],'.-')
    plt.semilogy(num_basis,np.abs(IP[t,:]),'-x')
    plt.legend(['|y - (Mx+q)|','|y - (Ax+q)|', '|(I-Pi)x|^2','<x,w>'])
    
    plt.show()
    
            
