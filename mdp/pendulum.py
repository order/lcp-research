import numpy as np
import state_remapper
import solvers.lqr as lqr

class PendulumRemapper(state_remapper.StateRemapper):
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.05)
        self.mass = kwargs.get('mass',1.0)
        self.gravity = 9.80665
        self.dampening = kwargs.get('dampening',0.1)
        self.length = kwargs.get('length',1.0)
        
    def remap(self,points,**kwargs):
        """
        Physics step for a double integrator:
        dx = Ax + Bu = [0 1; 0 0] [x;v] + [0; 1] u
        """
        if 1 == len(points.shape):
            points = points[np.newaxis,:]
            
        [N,d] = points.shape
        assert(d == 2)
        
        assert('action' in kwargs)
        u = kwargs['action']
        m = self.mass
        g = self.gravity
        b = self.dampening
        l = self.length
        
        ddt = 1/(m*l**2)*(u - m*g*l*np.sin(points[:,0]) - b*points[:,1])
        assert((N,) == ddt.shape)
        t_next = points[:,0] + self.step * points[:,1]
        dt_next = points[:,1] + self.step * ddt
        assert((N,) == t_next.shape)
        assert((N,) == dt_next.shape)
        
        ret = np.column_stack([t_next,dt_next])        
        assert((N,d) == ret.shape)
        
        return ret
        
    def forward_kinematics(self,state):
        """
        Converts the generalized coordinates into a list of 2D Cartesian positions
        """
        assert((2,) == state.shape)

        x0 = np.array([0,0]) # base always at origin
        
        # Position of middle joint
        x1 = self.length * np.array([np.sin(state[0]),-np.cos(state[0])])
        
        return np.column_stack([x0,x1]).T
        
def generate_lqr(pendulum,**kwargs):
    """
    Linearizes the pendulum dynamics at the top (pi,0) position
    
    Then generate the LQR controller
    """
    m = pendulum.mass
    L = pendulum.length
    g = pendulum.gravity
    b = pendulum.dampening
    
    I = m*(L**2) # Moment of inertia    
    
    A = np.array([[0,1],[m*g*L / I, -b / I]])
    B = np.array([[0,1]]).T
    x = np.array([np.pi,0]) # The set-point we're linearizing around

    assert((2,2) == A.shape)
    assert((2,1) == B.shape)

    Q = kwargs.get('Q',np.eye(2)) # State weights
    assert((2,2) == Q.shape)
    R = kwargs.get('R',np.array([[1e-3]])) # Control weights; must be matrix
    assert((1,1) == R.shape)
    
    K,S,E = lqr.lqr(A,B,Q,R)
    return (K,x)
  
