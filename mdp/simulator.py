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
        """
        This is where the heavy lifting is implemented
        """
        raise NotImplementedError()
        
    def simulate(self,state,policy,iters):  
        raise NotImplementedError()
        
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
        # Sanity checking
        assert(1 == len(self.state.shape)) # Vector
        
        # Get new positions
        poses = self.get_body_pos()       
        X = np.array([p[0] for p in poses])
        Y = np.array([p[1] for p in poses])
        self.plot_obj.set_data(X,Y)
        
        # Simulate forward one step
        actions = self.policy.get_decisions(self.state)
        assert((1,) == actions.shape)
        self.state = self.physics.remap(self.state,action=actions[0]).flatten()
        
        return self.plot_obj
        
    def simulate(self,state,policy,iters):  
        self.state = state
        self.policy = policy
        figure = plt.figure()
        ax = figure.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
        ax.grid()
        self.plot_obj, = ax.plot([],[],'o-',lw=2)
        anim = animation.FuncAnimation(figure, self.animate, iters, self.set_up, interval=2,repeat=False)
        plt.show()
        
class DoubleIntegratorSimulator(Simulator):
    def __init__(self,discretizer):
        self.discretizer = discretizer
        self.physics = discretizer.physics

    def set_up(self):
        return        

    def animate(self,frame_num):
        # Simulate forward one step
        actions = self.policy.get_decisions(self.state)
        assert((1,) == actions.shape)
        self.state = self.physics.remap(self.state,action=actions[0])
        
        # Add new data
        self.past_states = np.vstack([self.past_states,self.state])
        self.anim_obj.set_data(self.past_states[:,0],self.past_states[:,1])
        if actions[0] < 0:
            self.anim_obj.set_color('r')
        elif actions[0] > 0:
            self.anim_obj.set_color('b')
        else:
            self.anim_obj.set_color('g')
                
    def simulate(self,state,policy,iters): 
        if 1 == len(state.shape):
            state = state[np.newaxis,:]
        assert((1,2) == state.shape)
            
        self.state = state
        self.past_states = state
        self.policy = policy
        
        figure = plt.figure()
        ax = figure.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))
        ax.set_title('Double Integrator Animation')
        # Boundary
        [(x_lo,x_hi),(v_lo,v_hi)] = self.discretizer.get_basic_boundary()
        ax.plot([x_lo, x_hi, x_hi, x_lo,x_lo],[v_lo,v_lo,v_hi,v_hi,v_lo],'--k')
        
        # Animation line
        self.anim_obj, = ax.plot([x_lo, x_hi, x_hi, x_lo],[v_lo,v_lo,v_hi,v_hi],'-b',lw=2)
        
        # Fire off
        anim = animation.FuncAnimation(figure, self.animate, iters, self.set_up, interval=2,repeat=False)
        plt.show()

class AcrobotSimulator(Simulator):
    """
    Implements a ChainSimulator for the acrobot,
    which is an inverted pendulum with a
    motor in the middle joint
    """
    def __init__(self,discretizer):
        self.physics = discretizer.physics
        self.discretizer = discretizer

    def set_up(self):
        return
        
    def get_body_pos(self):
        assert((1,4) == self.state.shape)
        poses = self.physics.forward_kinematics(self.state.flatten())
        assert((3,2) == poses.shape)
        return poses  

    def animate(self,frame_num):
        # Sanity checking
        assert((1,4) == self.state.shape)
        
        # Get new positions
        poses = self.get_body_pos()       
        X = np.array([p[0] for p in poses])
        Y = np.array([p[1] for p in poses])
        self.anim_obj.set_data(X,Y)        
        
        # Simulate forward one step
        actions = self.policy.get_decisions(self.state)
        assert((1,) == actions.shape)
        self.state = self.physics.remap(self.state,action=actions[0])
        if actions[0] < 0:
            self.anim_obj.set_color('r')
        elif actions[0] > 0:
            self.anim_obj.set_color('b')
        else:
            self.anim_obj.set_color('g')
            
        # Add new data
        self.past_states = np.vstack([self.past_states,self.state])
        self.phase1_obj.set_data(self.past_states[:,0],self.past_states[:,2])
        self.phase2_obj.set_data(self.past_states[:,1],self.past_states[:,3])        
                
    def simulate(self,state,policy,iters):  
        if 1 == len(state.shape):
            state = state[np.newaxis,:]
        assert((1,4) == state.shape)
        self.state = state
        self.past_states = state
        self.policy = policy
        
        figure, axarr = plt.subplots(2, sharey=True)
        axarr[0].set_xlim(-5,5)
        axarr[0].set_ylim(-5,5)
        axarr[0].set_title('Animation')
        axarr[0].autoscale(False)
        axarr[0].set_aspect('equal')
        
        axarr[1].set_xlim(-5,5)
        axarr[1].set_ylim(-5,5)
        axarr[1].autoscale(False)
        axarr[1].set_aspect('equal')
        axarr[1].set_title('Phase Plot')
        
        self.anim_obj, = axarr[0].plot([],[],'o-',lw=2)
        self.phase1_obj,self.phase2_obj = axarr[1].plot([],[],'b-',[],[],'r-',lw=2,alpha=0.25)
        anim = animation.FuncAnimation(figure, self.animate, iters, self.set_up, interval=2,repeat=False)
        plt.show()
        
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
        [xf,yf] = [self.state[0,Idx[s]] for s in ['x','y']]
        (Xbs,Ybs) = self.physics.get_back_tire_pos(self.state)
        (xb,yb) = [q[0] for q in [Xbs,Ybs]]
        return [[xf,yf],[xb,yb]]
       
            
        
