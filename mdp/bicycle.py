from state_remapper import StateRemapper
from math import pi,pow

class BicycleRemapper(StateRemapper):
    """
    A state remapper for the Randlov and Alstrom bicycle.
    ("Learning to Drive a Bicycle using Reinforcement 
    Learning and Shaping", 1998)
    
    Much of the code is transliterated to python from the following C code:
    http://ai.fri.uni-lj.si/dorian/MLControl/BikeCCode.htm
    Note that some of this code is in "Danish". For example,
    constants governing the tires are annotated with "d", because
    the Danish word for tire is "daek".
    I kept this notation for debugging purposes.
    
    Much of the physics is unclear to me; would have been nice to see
    a derivation of this model and assumptions.
    Constant velocity seems a little unrealistic
    
    The parameterization is based on Lagoudakis and Parr
    ("Least-squares Policy Iteration",2003)
    """
    
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.01)
        
    def remap(self,points,**kwargs):
        (N,d) = points.shape
        assert(d == 9)
        # Dimensions of points are:
        # 0 theta;angle of handle bars
        # 1 dtheta; angular velocity of handle bars
        # 2 omega; angle of bike from vertical (lean)
        # 3 domega; angular velocity
        # 4 psi; heading angle (towards goal)
        # 5 xf; front tire x
        # 6 yf;
        # 7 xb;
        # 8 yb
        (theta,dtheta,omega,domega,psi,xf,yf,xb,yb) = range(9)
        X = points 
        # So the vector of thetas is X[:,theta]
        
        
        assert('action' in kwargs)
        u = kwargs['action']
        assert(u.shape == (2,))
        (tau,d) = (u[0],u[1])    
    
        dt = self.step
        v = 100.0/36.0 # fixed forward velocity
        g = 9.80665 # Were using 9.82. Seems like they were cycling roughly ~4.3km below sea-level.
        
        # Displacements
        dCM = 0.30 # distance of passenger's center of mass about seat?
        c = 0.66 # Probably NOT the speed of light
        h = 0.94 # height?
        l = 1.11 # Wheelhouse length (distance between front and back dropouts)
        R = 0.34 # Radius of tire
        
        # Masses (in kilograms):
        Mc = 15.0 # Total mass of cycle (including tires)
        Md = 1.7  # Mass of a tire 
        Mp = 60.0 # Mass of passenger
        M = Mc + Mp # Total system mass
        
        MaxHandleBar = 1.3963 # 80 degrees
        
        # Moments of inertia
        I_bike = (13.0/3.0) * Mc* pow(h,2) + Mp * pow(h+dCM,2)
        """
        Not sure, the last term is probably the passenger rotating along the bike at the ground
        First term is probably the inertia of the cycle rotating along the same; but where does
        13/3 come from?
        """
        I_dc = Md * pow(r,2) # Tire rotating around dropout
        I_dv = 1.5 * Md * pow(r,2) # Around contact with road (?)
        I_dl = 0.5 * Md * pow(r,2) # Around steering column
        dsigma = v / R # Tire angular velocity
        
        phi = X[:,omega] + np.atan(d / h) # Last term: component from lean
        assert(phi.shape = (N,))
        
        # Not sure what rCM is (nor rf or rb in original code)
        rCM = np.empty(N)
        mask = np.nonzero(X[:,theta])
        rCM[mask] = np.sqrt(pow(1-c,2) + pow(l,2) / np.pow(X[mask,theta],2))
        rf = 1 / np.abs(np.sin(X[:,theta]))
        rb = 1 / np.abs(np.tan(X[:,theta]))
        rCM[np.logical_not(mask)] = np.PINF
        rf[np.logical_not(mask)] = np.PINF
        rb[np.logical_not(mask)] = np.PINF
        
        # Blob of physics
        ddomega = (h*M*g*np.sin(phi)\
            - np.cos(phi) * (I_dc*dsigma*X[:,dtheta] + np.sign(X[:,theta])*pow(v,2)\
            *(Md*R*(1/rf + 1/rb) + M * h / rCM))) / I_bike
        assert(ddomega.shape = (N,))
        
        # Blob 2 of physics
        ddtheta = (tau - I_dv*domega*dsigma) / I_dl
        assert(ddtheta.shape = (N,))
        
        # Euler to update theta and omega (and velocities)
        X[:,domega] += ddomega * dt
        X[:,omega] += X[:,domega] * dt      
        X[:,dtheta] += ddtheta * dt
        X[:,theta] += X[:,dtheta] * dt
        X[:,theta] = np.amin(np.amax(X[:,theta],-MaxHandleBar),MaxHandleBar)
        
        # Update front tire position
        temp = (v*dt) / (2*rf)
        mask = temp > 1
        nmask = np.logical_not(mask) 
        temp[nmask] = np.sign(X[nmask,psi] + X[nmask,theta]) * np.asin(temp)
        temp[mask] = np.sign(X[mask,psi] + X[mask,theta]) * np.pi / 2
        assert(temp.shape = (N,))        
        X[:,xf] -= v*dt*np.sin(X[:,psi] + X[:,theta] + temp)
        X[:,yf] -= v*dt*np.cos(X[:,psi] + X[:,theta] + temp)
        
        # Update back tire position
        temp = (v*dt) / (2*rb)
        mask = temp > 1
        nmask = np.logical_not(mask) 
        temp[nmask] = np.sign(X[nmask,psi]) * np.asin(temp)
        temp[mask] = np.sign(X[mask,psi]) * np.pi / 2
        assert(temp.shape = (N,))
        
        X[:,xb] -= v*dt*np.sin(X[:,psi] + temp)
        X[:,yb] -= v*dt*np.cos(X[:,psi] + temp)
        
        # Enforce contraint that bike length is l (lowercase L)
        pos_diff = X[:,(xb,yb)] - X[:,(xf,yf)]
        (x_diff,y_diff) = (0,1)
        bike_norm = np.linalg.norm(pos_diff,axis=1)
        X[:,xb] += (X[:,xb] - X[:,xf]) * (l - bike_norm) / bike_norm
        X[:,yb] += (X[:,yb] - X[:,yf]) * (l - bike_norm) / bike_norm
        
        # Update psi (heading angle)
        # This a blob trying to vectorize an (if,elif,else) block in the original code
        if_mask = np.logical_and(pos_diff[:,x_diff] == 0, pos_diff[:,y_diff] < 0)
        X[if_mask,psi] = np.pi        
        elif_mask = np.logical_and(np.logical_not(if_mask), pos_diff[:,y_diff] > 0)
        X[elif_mask,psi] = np.atan(pos_diff[:,x_diff] / pos_diff[:,y_diff])
        else_mask = np.logical_and(np.logical_not(if_mask), np.logical_not(elif_mask))
        X[else_mask,psi] = np.sign(pos_diff[:,x_diff]) * (np.pi / 2) \
            - np.atan(pos[:,y_diff] / pos[:,x_diff])
        
        return X