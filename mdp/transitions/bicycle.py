from state_remapper import StateRemapper
from math import pi,pow
import numpy as np

class struct():
    pass

class BicycleRemapper(StateRemapper):
    """
    A state remapper for the Randlov and Alstrom bicycle.
    ("Learning to Drive a Bicycle using Reinforcement 
    Learning and Shaping", 1998)
    
    Much of the code is transliterated to python from the following C code:
    http://ai.fri.uni-lj.si/dorian/MLControl/BikeCCode.htm".
    I kept this notation for clarity.
    
    Much of the physics is unclear to me; would have been nice to see
    more of a derivation of this model and assumptions.
    Constant velocity seems a little unrealistic.
    
    The parameterization is based on Lagoudakis and Parr
    # ("Least-squares Policy Iteration",2003)
    """
    
    def __init__(self,**kwargs):
        self.step = kwargs.get('step',0.01)       
        dim_names = ['theta','dtheta','omega','domega','psi','x','y']
        self.dim_ids = dict(zip(dim_names,range(len(dim_names))))
                
        """
        Loose constants. Read from kwargs w/ default?
        """  
        params = struct()
        params.dt = self.step # Simulation step
        params.v = 100.0/36.0 # fixed forward velocity
        params.g = 9.80665 # Were using 9.82. Seems like they were cycling roughly ~4.3km below sea-level.
        
        # Displacements
        params.dCM = 0.30 # Distance between cyclist CM and bike CM
        params.c = 0.66 # Distance of system CM from front dropout
        params.h = 0.94 # Height of CM above ground
        params.l = 1.11 # Wheelhouse length (distance between front and back dropouts)
        params.R = 0.34 # Radius of tire
        
        # Masses (in kilograms):
        params.Mc = 15.0 # Total mass of cycle (including tires)
        params.Md = 1.7  # Mass of a tire 
        params.Mp = 60.0 # Mass of passenger
        params.M = params.Mc + params.Mp # Total system mass
        
        params.MaxHandleBar = 1.3963 # 80 degrees
        
        # Moments of inertia
        params.I_bike = (13.0/3.0) * params.Mc* pow(params.h,2) + params.Mp * pow(params.h+params.dCM,2)
        """
        Not sure about this formula.
        The last term is probably the passenger rotating along the bike at the ground
        First term is probably the inertia of the cycle rotating along the same; but where does
        13/3 come from?
        """
        params.I_dc = params.Md * pow(params.R,2) # Tire rotating around dropout
        params.I_dv = 1.5 * params.Md * pow(params.R,2) # Around contact with road 
        params.I_dl = 0.5 * params.Md * pow(params.R,2) # Around steering column
        params.dsigma = params.v / params.R # Tire angular velocity
        
        self.params = params
        
    def get_back_tire_pos(self,points):
        x = self.dim_ids['x']
        y = self.dim_ids['y']
        psi = self.dim_ids['psi']
        l = self.params.l
        
        Xs = points[:,x] + l * np.sin(points[:,psi])
        Ys = points[:,y] + l * np.cos(points[:,psi])
        return (Xs,Ys)

    def remap(self,points,**kwargs):
        (N,d) = points.shape
        S = len(self.dim_ids)
        assert(d == S)
        [theta,dtheta,omega,domega,psi,x,y] = range(S)
        
        dt = self.params.dt
        v = self.params.v
        g = self.params.g
        dCM = self.params.dCM
        c = self.params.c
        h = self.params.h
        l = self.params.l
        R = self.params.R    
        Mc = self.params.Mc
        Md = self.params.Md
        Mp = self.params.Mp
        M = self.params.M  
        MaxHandleBar = self.params.MaxHandleBar
        I_bike = self.params.I_bike
        I_dc = self.params.I_dc
        I_dv = self.params.I_dv
        I_dl = self.params.I_dl
        dsigma = self.params.dsigma
        
        X = points # Shorthand
        
        assert('action' in kwargs)
        u = kwargs['action']
        assert(u.shape == (2,))
        (tau,d) = (u[0],u[1]) # Handlebar torque and center of mass 'lean'
        
        # Total tilt of the center of mass; omega + leaning component
        phi = X[:,omega] + np.arctan(d / h)
        assert((N,) == phi.shape)
        
        # Turning model. 
        rf = np.empty(N) # Distance of front tire for "instant center"
        rb = np.empty(N) # For back tire
        rCM = np.empty(N) # For center of mass
        
        zmask = (0 == X[:,theta])
        mask = np.logical_not(zmask)

        # Radii of circular paths for front tire, back tire, and center of mass
        rf[mask] = l / np.abs(np.sin(X[mask,theta]))
        rb[mask] = l / np.abs(np.tan(X[mask,theta]))        
        rCM[mask] = np.sqrt(pow(l - c,2) + pow(l,2) / np.power(np.tan(X[mask,theta]),2))
        
        # When handlebars are straight, there is no instance center
        rf[zmask] = np.PINF
        rb[zmask] = np.PINF
        rCM[zmask] = np.PINF
        
        # Blob of physics governing angular acceleration of tilt
        ddomega = (h*M*g*np.sin(phi)\
            - np.cos(phi) * (I_dc*dsigma*X[:,dtheta] + np.sign(X[:,theta])*pow(v,2)\
            *(Md*R*(1/rf + 1/rb) + M * h / rCM))) / I_bike
        assert((N,) == ddomega.shape)
        
        # Blob 2 of physics governming angular acceleration of handle bars
        # Torque resisted by a gyroscopic term
        ddtheta = (tau - I_dv*X[:,domega]*dsigma) / I_dl
        assert((N,) == ddomega.shape)
        
        # Euler to update theta and omega (and velocities)
        X[:,domega] += ddomega * dt
        X[:,omega] += X[:,domega] * dt      
        X[:,dtheta] += ddtheta * dt
        X[:,theta] += X[:,dtheta] * dt
        X[X[:,theta] > MaxHandleBar,theta] = MaxHandleBar
        X[X[:,theta] < -MaxHandleBar,theta] = -MaxHandleBar
        
        # Get the back tire position from psi and front tire position
        (x_back,y_back) = self.get_back_tire_pos(X)
            
        # Update front tire position
        temp = (v*dt) / (2.0*rf)
        mask = temp > 1
        nmask = np.logical_not(mask) 
        temp[nmask] = np.sign(X[nmask,psi] + X[nmask,theta]) * np.arcsin(temp)
        temp[mask] = np.sign(X[mask,psi] + X[mask,theta]) * np.pi / 2.0
        assert((N,) == temp.shape)        
        X[:,x] -= v*dt*np.sin(X[:,psi] + X[:,theta] + temp)
        X[:,y] -= v*dt*np.cos(X[:,psi] + X[:,theta] + temp)
        
        # Update back tire position
        temp = (v*dt) / (2.0*rb)
        mask = temp > 1
        nmask = np.logical_not(mask) 
        temp[nmask] = np.sign(X[nmask,psi]) * np.arcsin(temp)
        temp[mask] = np.sign(X[mask,psi]) * np.pi / 2.0
        assert((N,) == temp.shape)
        
        x_back -= v*dt*np.sin(X[:,psi] + temp)
        y_back -= v*dt*np.cos(X[:,psi] + temp)
        
        pos_diff = np.column_stack([x_back,y_back]) - X[:,(x,y)]
        (x_diff,y_diff) = (0,1)
            
        # Update psi (heading angle)
        # This a blob trying to vectorize an (if,elif,else) block in the original code
        if_mask = np.logical_and(pos_diff[:,x_diff] == 0, pos_diff[:,y_diff] < 0)
        X[if_mask,psi] = np.pi        
        elif_mask = np.logical_and(np.logical_not(if_mask), pos_diff[:,y_diff] > 0)
        X[elif_mask,psi] = np.arctan(pos_diff[:,x_diff] / pos_diff[:,y_diff])
        else_mask = np.logical_and(np.logical_not(if_mask), np.logical_not(elif_mask))
        X[else_mask,psi] = np.sign(pos_diff[:,x_diff]) * (np.pi / 2) \
            - np.arctan(pos_diff[:,y_diff] / pos_diff[:,x_diff])
       
        return X