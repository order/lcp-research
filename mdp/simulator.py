import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

class Simulator(object):
    """
    Generic simulation object
    """
    def set_up(self):
        raise NotImplementedError()  
        
    def animate(self,frame_num):
        raise NotImplementedError()
        
    def simulate(self,state,action,iters):  
        self.state = state
        self.action = action
        figure = plt.figure()
        ax = figure.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
        ax.grid()
        self.plot_obj, = ax.plot([],[],'o-',lw=2)
        anim = animation.FuncAnimation(figure, self.animate, iters, self.set_up, interval=2)
        plt.show()
        
class ChainSimulator(Simulator):
    """
    Assumes the object to be simulated is a 2D chain
    Implements everything except for "get_body_pos" which
    should be the (x,y) positions of each point in the chain
    """
    def set_up(self):
        return
        
    def get_body_pos(self):
        raise NotImplementedError()
        
    def animate(self,frame_num):
        # Get new positions
        poses = self.get_body_pos()       
        X = np.array([p[0] for p in poses])
        Y = np.array([p[1] for p in poses])
        self.plot_obj.set_data(X,Y)
        
        # Simulate forward one step
        self.state = self.physics.remap(self.state,action=self.action)
        
        return self.plot_obj

class AcrobotSimulator(ChainSimulator):
    """
    Implements a ChainSimulator for the acrobot,
    which is an inverted pendulum with a
    motor in the middle joint
    """
    def __init__(self,physics):
        self.physics = physics

    def get_body_pos(self):
        assert((4,) == self.state.shape)
        q = self.state[:2] # Joint angles
        poses = self.physics.forward_kinematics(q)
        assert((3,2) == poses.shape)
        return poses    
        
class BicycleSimulator(ChainSimulator):
    """
    Simulator for the Randlov-Astrom bike. Just two points
    """    
    def __init__(self,physics):
        self.physics = physics
        
    def get_body_pos(self):
        N = self.state.size
        Idx = self.physics.dim_ids
        assert len(Idx) == N
        [xf,yf,xb,yb] = [self.state[0,Idx[s]] for s in ['xf','yf','xb','yb']]
        return [[xf,yf],[xb,yb]]
       
            
        
