import math
import numpy as np
import state_remapper

class AcrobotRemapper(state_remapper.StateRemapper):
    """
    Physics remapper for the acrobot. Using the varient in
    Tedrake's notes.
    x
     \
      \
       o----o
    """
    
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.05)
        self.l1 = kwargs.get('l1',1.0)
        self.l2 = kwargs.get('l2',1.0)
        self.m1 = kwargs.get('m1',1.0)
        self.m2 = kwargs.get('m2',1.0)
        self.lc1 = self.l1 / 2.0
        self.lc2 = self.l2 / 2.0
        # Momements of inertia; taken from pivots
        self.I1 = (self.m1 * self.l1**2) # Solid rod
        self.I2 = (self.m2 * self.l2**2)
        self.g = 9.80665
        self.dampening = 0.1

    def get_HCGB(self,q,dq,u):
        """
        These are the four matrices involved in the 
        equations of motion:
        H*ddq + C*dq + G = B*u
        """
        assert((2,) == q.shape)
        assert((2,) == dq.shape)
        
        l1 = self.l1
        l2 = self.l2
        lc1 = self.lc1
        lc2 = self.lc2
        
        m1 = self.m1
        m2 = self.m2
        
        I1 = self.I1
        I2 = self.I2
        
        g = self.g
        d = self.dampening
        
        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s12 = np.sin(q[0]+q[1])

        temp1 = m2*g*l2
        temp2 = m2*l1*lc2
        
        H_11 = I1 + I2 + m2*l1*l1 + 2.0*temp2*c2
        H_12 = I2 + temp2*c2
        H_21 = H_12
        H_22 = I2
        H = np.array([[H_11,H_12],[H_21,H_22]])

        # Todo: add dampening
        C_11 = -2.0*temp2*s2*dq[1] + d
        C_12 = -temp2*s2*dq[1]
        C_21 = temp2*s2*dq[0]
        C_22 = d
        C = np.array([[C_11,C_12],[C_21,C_22]])
        
        G_1 = (m1*lc1 + m2*l1)*g*s1 +temp1*s12
        G_2 = temp1*s12
        G = np.array([G_1,G_2])

        B = np.array([0,1])

        return (H,C,G,B)
        
    def get_acc(self,q,dq,u):
        (H,C,G,B) = self.get_HCGB(q,dq,u)
        b = B.dot(u) - C.dot(dq) - G
        assert((2,) == b.shape)
        assert((2,2) == H.shape)
        
        #ddq = np.linalg.solve(H,b)
        detH = H[0,0]*H[1,1] - H[1,0]*H[0,1]
        ddq1 = (H[1,1]*b[0] - H[0,1]*b[1]) / detH
        ddq2 = (-H[1,0]*b[0] + H[0,0] * b[1]) / detH
        ddq = np.array([ddq1,ddq2])
        assert((2,) == ddq.shape)        
        return ddq
        
    def remap(self,points,**kwargs):
        """
        Physics step for the acrobot
        """
        if 1 == len(points.shape):
            points = points[np.newaxis,:]            
        
        [N,d] = points.shape
        assert(d == 4)
        
        assert('action' in kwargs)
        u = kwargs['action'] # Torque for the middle joint
        if hasattr(u,'shape'):
            assert((1,) == u.shape)
            u = u[0]
        
        new_points = np.empty((N,d))
        for state_id in xrange(N):
            q = points[state_id,:2].flatten() # The two angles
            dq = points[state_id,2:].flatten() # Angular velocities
        
            ddq = self.get_acc(q,dq,u)
            assert((2,) == ddq.shape)
            dq += self.step*ddq
            assert((2,) == dq.shape)
            q += self.step*dq
            assert((2,) == q.shape)
            new_points[state_id,:2] = q
            new_points[state_id,2:] = dq
            
        return new_points
        
    def forward_kinematics(self,state):
        """
        Converts the generalized coordinates into a list of 2D Cartesian positions
        """
        assert((4,) == state.shape)
        q = state[:2]
        assert(q.size == 2)

        x0 = np.array([0,0]) # base always at origin
        
        # Position of middle joint
        x1 = self.l1 * np.array([np.sin(q[0]),-np.cos(q[0])])
        theta_12 = np.sum(q)
        
        # Position of end joint
        x2 = x1 + self.l2 * np.array([np.sin(theta_12),-np.cos(theta_12)])
        return np.column_stack([x0,x1,x2]).T
