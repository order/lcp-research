import numpy as np
import matplotlib.pyplot as plt

def next_dubbins_state(s,u,h):
    """
    Forward Euler for Dubbins dynamics
    @param s: state, Nx3
    @param u: turn rate
    @param h: Euler constant (time step)
    @return: new states
    """
    if 1 == len(s.shape) and 3 == s.size:
        s = s.reshape(1,3)
    N,D = s.shape
    assert 3 == D

    dx = np.cos(s[:,2])
    dy = np.sin(s[:,2])
    dt = u*np.ones(N)

    ds = np.array([dx,dy,dt]).T
    assert ds.shape == s.shape

    return s + h*ds

def next_relative_state(s,u,v,h):

def single_test():
    T = 1000
    states = np.zeros((T,3))
    
    for i in xrange(T-1):
        states[i+1,:] = next_dubbin_state(states[i,:],0.1,0.05)

    plt.plot(states[:,0], states[:,1])
    plt.show()

def dual_test():
    T = 1000
    h = 0.05
    states = np.zeros((T,6))
    states[0,3:] = np.array([10,10,np.pi])
    
    for i in xrange(T-1):
        states[i+1,:3] = next_dubbin_state(states[i,:3],0.1,h)
        states[i+1,3:] = next_dubbin_state(states[i,3:],-0.05,h)

    plt.figure()
    plt.plot(states[:,0], states[:,1])
    plt.plot(states[:,3], states[:,4])

def relative_test():
    T = 1000
    h = 0.05
    states = np.zeros((T,6))
    states[0,3:] = np.array([10,10,np.pi])
    
    for i in xrange(T-1):
        states[i+1,:3] = next_dubbin_state(states[i,:3],0.1,h)
        states[i+1,3:] = next_dubbin_state(states[i,3:],-0.05,h)

    plt.figure()
    plt.plot(states[:,0] - states[:,3], states[:,1] - states[:,4])
        
if __name__ == "__main__":
    dual_test()
    relative_test()
    plt.show()
    
