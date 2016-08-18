import numpy as np
import scipy.sparse as sps
import scipy.signal as signal

from lcp import *
from config.solver import *

from experiment import *

import matplotlib.pyplot as plt

def build_aug_plcp(P,lcp):
    U = P.T.dot(lcp.M)
    q = P.dot(P.T.dot(lcp.q))
    
    plcp = ProjectiveLCPObj(P,U,U,q)
    scale = 10
    (aplcp,x0,y0,w0) = augment_plcp(plcp,scale)
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

def ray(a):
    a = a / np.linalg.norm(a)
    return np.hstack([np.zeros((2,1)),a[:,np.newaxis]])

if __name__ == "__main__":
    Trials = 256
    Dim = 2

    
    #a = np.random.rand(Dim)
    #b = a + 1
    #b = np.random.rand(Dim)
    a = np.array([0.6,0.2])
    b = np.array([0.2,0.6])
    
    c = np.array([1.0,1.0])
    c = c / np.linalg.norm(c)
    
    lcp = build_max_lcp(a,b,c)
    (exact_p,exact_d,exact_data) = kojima_solve(lcp)
    ans = (exact_p[:2],exact_p[2:4],exact_p[4:])
    v = ans[0]
    
    theta = np.linspace(0,np.pi,Trials)
    Basis = np.array([np.sin(theta),np.cos(theta)])  
    Error = np.empty((3,Trials))

    #plt.rc('text',usetex=True)
    
    for t in xrange(Trials):
        B = [(Basis[:,t])[:,np.newaxis],sps.eye(Dim), sps.eye(Dim)]
        for blk in xrange(3): # Blocks
            P = sps.block_diag(B)
            
            (plcp,aplcp,x0,y0,w0) = build_aug_plcp(P,lcp)
            (aug_p,aug_d,aug_data) = projective_solve(aplcp,x0,y0,w0)
            
            (res_p,res_d) = (aug_p[-1],aug_d[-1])
            print res_p,res_d
            if np.abs(res_p) > 1e-6 or np.abs(res_d) > 1e-6:
                Error[blk,t] = np.nan
                continue

            (p,d) = (aug_p[:-1],aug_d[:-1])
            approx_ans = p[:2]
            Error[blk,t] = np.log(np.linalg.norm(approx_ans - v))

            B = B[1:] + B[:1] # Rotate

    print 'a',a
    print 'b',b
    print 'min(a,b)', np.minimum(a,b)

    A = ray(a)
    B = ray(b)
    V = ray(ans[0])
    F1 = ray(ans[1])
    F2 = ray(ans[2])
    plt.figure()
    for blk in xrange(3):        
        plt.subplot(1,3,blk+1)

        for X in [V,F1,F2]:
            plt.plot(X[0,:],X[1,:])
        plt.legend(['V','F1','F2'],loc='upper left')
        
        plt.scatter(Basis[0,:],Basis[1,:],c=Error[blk,:],
                    s=25,lw=0,alpha=0.5)
        plt.scatter(-Basis[0,:],-Basis[1,:],c=Error[blk,:],
                    s=25,lw=0,alpha=0.5)
        plt.colorbar()
    plt.show()
