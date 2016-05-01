import numpy as np
from math import pi,sqrt
import matplotlib.pyplot as plt
from state_remapper import StateRemapper

import types

import scipy.stats
from utils.parsers import KwargParser

class HillcarRemapper(StateRemapper):
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('mass',1.0,[int,float])
        parser.add('step',0.01,float)
        parser.add('slope',basic_slope,types.FunctionType)
        args = parser.parse(kwargs)
        
        self.g = 9.806
        self.mass = args['mass']
        self.step = args['step']
        self.slope = args['slope']
        
    def remap(self,points,**kwargs):
        """
        Physics step for hill car based on slope.
        """
        [N,d] = points.shape
        assert(d == 2)
        
        Mass = kwargs.get('mass',1)
        G = kwargs.get('gravity',9.806)
        step = kwargs.get('step',0.01)
        
        u = kwargs['action']
        
        theta = np.arctan(self.slope(points[:,0]))
        F_grav = -Mass*G*np.sin(theta)
        F_cont = u*np.cos(theta)
        
        x_next = points[:,0] + step * points[:,1]
        v_next = points[:,1] + step * (F_grav + F_cont)
        return np.column_stack((x_next,v_next))
    
# Simple two hill terrain
def sample_hill_slope(x):
    """
    A sample two hill terrain slope
    """
    a1 = 0.5
    mu1 = -1.25
    sigma1 = 0.25
    a2 = 1
    mu2 = 1.1
    sigma2 = 0.85
    return a1*derv_normpdf(x,mu1,sigma1) + a2*derv_normpdf(x,mu2,sigma2)

def normpdf(x,mu,sigma):
    return scipy.stats.norm.pdf(x,loc=mu,scale=sigma)

def derv_normpdf(x,mu,sigma):
    """
    Derivative of the normal function
    """
    return (mu - x) / (sqrt(2*pi) * sigma**3) \
        * np.exp(-(x - mu)**2 / (2*sigma**2))

def basic_slope(x,**kwargs):
    """
    Basic slope; looks like a big 'W' with smoothed transitions
    """
    parser = KwargParser()
    parser.add('grade',0.15,[int,float])
    parser.add('bowl',1.0,[int,float])
    parser.add('hill',5.0,[int,float])
    args = parser.parse(kwargs)

    grade = args['grade']
    bowl = args['bowl']
    hill = args['hill']
    
    assert(1 == len(x.shape))
    theta = np.zeros(x.shape)

    mask = np.abs(x) <= bowl
    theta[mask] = -grade/bowl*x[mask]

    mask = np.logical_and(np.abs(x) > bowl,np.abs(x) <= bowl+hill)
    theta[mask] = -grade * np.sign(x[mask])

    mask = np.logical_and(np.abs(x) > bowl+hill,np.abs(x) <= 3*bowl+hill)
    theta[mask] = np.sign(x[mask])*\
                  (grade/bowl*(np.abs(x[mask]) - bowl - hill) - grade)

    mask = np.abs(x) > 3*bowl+hill
    theta[mask] = grade *np.sign(x[mask])
    
    return theta
