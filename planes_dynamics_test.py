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
    if 1 == len(s.shape) and 3 == s.size:
        s = s.reshape(1,3)
    x = s[:,0]
    y = s[:,1]
    t = s[:,2]

    # Translate to zero out own-ship (x,y)
    x_p = x + h*(np.cos(t) - 1)
    y_p = y + h*(np.sin(t))

    # Rotation to zero out own-ship rotation
    phi = -h * u
    
    x_n = np.cos(phi)*x_p - np.sin(phi)*y_p
    y_n = np.sin(phi)*x_p + np.cos(phi)*y_p 
    t_n = t + h*(v - u)

    return np.array([x_n,y_n,t_n]).T

def single_test(T,h):
    states = np.zeros((T,3))
    
    for i in xrange(T-1):
        states[i+1,:] = next_dubbins_state(states[i,:],0.1,h)

    plt.plot(states[:,0], states[:,1])
    plt.show()

def dual_test(T,h):
    states = np.zeros((T,6))
    states[0,3:] = np.array([10,10,np.pi])
    
    for i in xrange(T-1):
        states[i+1,:3] = next_dubbins_state(states[i,:3],0.1,h)
        states[i+1,3:] = next_dubbins_state(states[i,3:],-0.05,h)

    plt.figure()
    plt.plot(states[:,0], states[:,1])
    plt.plot(states[:,3], states[:,4])
    plt.figure()
    rx = states[:,3] - states[:,0]
    ry = states[:,4] - states[:,1]
    plt.plot(np.sqrt(np.power(rx,2) + np.power(ry,2)))
    return states

def relative_test(T,h):
    states = np.zeros((T,3))
    states[0,:] = np.array([10,10,np.pi])
    
    for i in xrange(T-1):
        states[i+1,:] = next_relative_state(states[i,:],0.1,-0.05,h)

        
    plt.figure()
    plt.plot(states[:,0], states[:,1])
    plt.figure()
    plt.plot(np.sqrt(np.power(states[:,0],2) + np.power(states[:,1],2)))
    return states
        
if __name__ == "__main__":
    T =5000
    h=0.05
    dual_test(T,h)
    relative_test(T,h)

    plt.show()
    
