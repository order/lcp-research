import numpy as np

def rand_spsd_lcp(N,**kwargs):
    A = 2*np.random.rand(N,N)-1
    M = np.dot(A.T, A)
    q = 2*np.random.rand(N)-1
    return (M,q)
    
def rand_psd_lcp(N,**kwargs):
    A = 2*np.random.rand(N,N)-1
    B = 2*np.random.rand(N,N)-1
    M = np.dot(A.T, A) + 0.5*(B - B.T)
    q = 2*np.random.rand(N)-1
    return (M,q)

def rand_lpish(N,m,**kwargs):
    assert(N > m)
    A = 2*np.random.rand(N-m,m)-1
    BigZero = np.zeros((N-m,N-m))
    LittleZero = np.zeros((m,m))
    Top = np.hstack((BigZero, A))
    Bottom = np.hstack((-A.T, LittleZero))
    M = np.vstack((Top,Bottom))
    
    q = 2*np.random.rand(N)-1
    return (M,q)
