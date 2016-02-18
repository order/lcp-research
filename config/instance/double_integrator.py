import numpy as np
import config
import config.instance.gens.double_integrator_gen as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class DoubleIntegratorConfig(config.InstanceConfig):
    def __init__(self):
        params = {}
        params['x_desc'] = (-5,5,30)
        params['v_desc'] = (-4,4,20)
        params['a_desc'] = (-1,1,3)
        params['dampening'] = 1e-5
        
        center = np.zeros(2)
        radius = 0.15
        params['cost_obj'] = mdp.BallSetFn(center,radius)
        
        params['discount'] = 0.997
        self.params = params

        self.gen_fn = gen.DoubleIntegratorGenerator(**self.params)
