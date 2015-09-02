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
        self.I1 = (self.m1 * self.l1**2)
        self.I2 = (self.m2 * self.l2**2) 
        self.g = 9.80665
        
    def get_acc(self,q,dq,u):
        l1 = self.l1
        l2 = self.l2
        lc1 = self.lc1
        lc2 = self.lc2
        m1 = self.m1
        m2 = self.m2
        I1 = self.I1
        I2 = self.I2
        g = self.g

        c1 = np.cos(q[0] - math.pi/2)
        c2 = np.cos(q[1])
        s2 = np.sin(q[1])
        c12 = np.cos(q[0]+q[1] - math.pi/2)


        h_11 = m1*pow(lc1,2) + m2 *(pow(l1,2) + pow(lc2,2) + 2.0 * l1*lc2*c2) + I1 + I2
        h_22 = m2*pow(lc2,2) + I2
        h_12 = m2*(pow(lc2,2) + l1*lc2*c2) + I2
        h_21 = h_12
        
        c_1 = -m2*l1*lc2*(dq[1]**2)*s2 - 2.0*m2*l1*lc2*dq[0]*dq[1]*s2
        c_2 = m2*l1*lc2*(dq[0]**2)*s2
        
        g_1 = (m1*lc1 + m2*l1)*g*c1 + m2*lc2*g*c12
        g_2 = m2*lc2*g*c12
        
        ddq_2 = (h_11 * (u - c_2 - g_2) + h_12*(c_1 + g_1)) / (h_11 * h_22 - h_12*h_21)
        ddq_1 = -(h_12*ddq_2 + c_1 + g_1) / h_11
        
        return np.array([ddq_1,ddq_2])
        
    def remap(self,points,**kwargs):
        """
        Physics step for the acrobot
        """
        [N,d] = points.shape
        assert(d == 4)
        
        assert('action' in kwargs)
        u = kwargs['action'] # Torque for the middle joint
        
        new_points = np.empty((N,d))
        for state_id in xrange(N):
            q = points[state_id,:2].flatten() # The two angles of the acrobot
            dq = points[state_id,2:].flatten() # Angular velocities
        
            ddq = self.get_acc(q,dq,u)
            dq += self.step*ddq
            q += self.step*dq
            new_points[state_id,:2] = q
            new_points[state_id,2:] = dq
            
        return new_points
        
    def forward_kinematics(self,X):
        """
        Converts the generalized coordinates into a list of 2D Cartesian positions
        """
        q = (X.flatten())[:(X.size/2)]
        assert(q.size == 2)

        x0 = np.array([0,0])
        
        # Position of middle joint
        x1 = self.l1 * np.array([np.sin(q[0]),-np.cos(q[0])])
        theta_12 = np.sum(q)
        
        # Position of end joint
        x2 = x1 + self.l2 * np.array([np.sin(theta_12),-np.cos(theta_12)])
        return [x0,x1,x2]