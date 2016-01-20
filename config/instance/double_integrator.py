import numpy as np
from config.instance.gens.double_integrator_gen import DoubleIntegratorGenerator

class Config(object):
    def __init__(self):
        params['name'] = 'Small Double Integrator'

        params['x_desc'] = (-5,5,20)
        params['v_desc'] = (-6,6,20)
        params['a_desc'] = (-1,1,3)
        
        center = np.zeros(2)
        radius = 0.25
        params['cost_obj'] = mdp.BallSetFn(center,radius)
        
        params['discount'] = 0.99
        self.params = params

    def get_object(self):
        return DoubleItegratorGenerator(**self.params)
