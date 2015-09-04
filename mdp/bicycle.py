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
    
    The parameterization is based on Lagoudakis and Parr
    ("Least-squares Policy Iteration",2003)
    """
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.01)
        
    def remap(self,points,**kwargs):
        (N,d) = points.shape
        assert(d == 5)
        # Dimensions of points are:
        # 0 theta;angle of handle bars
        # 1 dtheta; angular velocity of handle bars
        # 2 omega; angle of bike from vertical (lean)
        # 3 domega; angular velocity
        # 4 psi; heading angle (towards goal)
        (theta,dtheta,omega,domega,psi) = range(5) # 
        
        assert('action' in kwargs)
        u = kwargs['action']
        assert(u.shape == (2,))
        (tau,d) = (u[0],u[1])    
    
        dt = self.step
        v = 100.0/36.0 # fixed forward velocity
        g = 9.80665
        
        # Displacements
        dCM = 0.30 # distance of passenger's center of mass about seat?
        c = 0.66 # Probably NOT the speed of light
        h = 0.94 # height?
        l = 1.11 # Wheelhouse length (distance between front and back dropouts)
        R = 0.34 # Radius of tire
        
        # Masses (in kilograms):
        Mc = 15.0 # Mass of... cycle?
        Md = 1.7  # Mass of.... tire? 
        Mp = 60.0 # Mass of... passenger?
        M = Mc + Mp # Total system mass?
        
        # Moments of inertia
        I_bike = (13.0/3.0) * Mc* pow(h,2) + Mp * pow(h+dCM,2)
        """
        Not sure, the last term is probably the passenger rotating along the bike at the ground
        First term is probably the inertia of the cycle rotating along the same; but where does
        13/3 come from?
        """
        # Tires rotating various axes? Apparently the danish word for tire is "daek"
        I_dc = Md * pow(r,2) # Around dropout
        I_dv = 1.5 * Md * pow(r,2) # Around contact with road?
        I_dl = 0.5 * Md * pow(r,2) # Around steering column
        dsigma = v / R # Tire angular velocity
        
        X = points # Readability        
        phi = X[:,omega] + np.atan(d / h) # Last term: component from lean
        assert(phi.shape = (N,))
        
        # Not sure what rCM is (nor rf or rb in original code)
        rCM = np.empty(N)
        mask = np.nonzero(X[:,theta])
        rCM[mask] = np.sqrt(pow(1-c,2) + pow(l,2) / np.pow(X[mask,theta],2))
        rCM[np.logical_not(mask)] = np.PINF
        inv_rf = np.abs(np.sin(X[:,theta]))
        inv_rb = np.abs(np.tan(X[:,theta]))
        # Original code inverted rf and rb just to "revert" them.
        
        # Blob of physics
        ddomega = (h*M*g*np.sin(phi)\
            - np.cos(phi) * (I_dc*dsigma*X[:,dtheta] + np.sign(X[:,theta])*pow(v,2)\
            *(Md*R*(inv_rf + inv_rb) + M * h / rCM))) / I_bike
        assert(ddomega.shape = (N,))
        
        # Blob 2 of physics
        ddtheta = (tau - I_dv*domega*dsigma) / I_dl
        assert(ddtheta.shape = (N,))
        
        # Euler
        X[:,domega] += ddomega * dt
        X[:,omega] += X[:,domega] * dt      
        X[:,dtheta] += ddtheta * dt
        X[:,theta] += X[:,dtheta] * dt        
        