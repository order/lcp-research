import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Simulator(object):
    def simulate(self,start,physics,iters):
        raise NotImplementedError()
        
class ChainSimulator(object):
    def __init__(self,dim,physics):
        assert(dim == 2) # Extend to 3d later
        self.dim = dim
        
        self.physics = physics
        
    def set_up(self):
        return self.plot_obj
        
    def get_q(self):
        N = self.state.size
        assert(0 == N % 2)
        q = self.state[:(N/2)]
        assert(0 == q.size % self.dim)
        return q
        
    def animate(self,frame_num):
        # Get new positions
        q = self.get_q()        
        poses = self.physics.forward_kinematics(q)
        X = np.array([p[0] for p in poses])
        Y = np.array([p[1] for p in poses])
        self.plot_obj.set_data(X,Y)
        
        # Simulate forward one step
        self.state = self.physics.remap(self.state,action=0)
        
        return self.plot_obj                
        
    def simulate(self,start,iters):        
        self.state = start
        
        # Set up figures and what not
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
        ax.grid()
        self.plot_obj, = ax.plot([],[],'o-',lw=2)
        
        anim = animation.FuncAnimation(fig, self.animate, iters, self.set_up, interval=2)
        plt.show()