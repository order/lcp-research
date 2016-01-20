import numpy as np
import config
import config.instance.gens.double_integrator_gen as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class SmallDoubleIntegratorConfig(config.Config):
    def __init__(self):
        params = {}
        params['x_desc'] = (-5,5,20)
        params['v_desc'] = (-6,6,20)
        params['a_desc'] = (-1,1,3)
        
        center = np.zeros(2)
        radius = 0.25
        params['cost_obj'] = mdp.BallSetFn(center,radius)
        
        params['discount'] = 0.99
        self.params = params

    def build(self):
        return gen.DoubleIntegratorGenerator(**self.params)
