import numpy as np
import config
import config.instance.gens.hallway_continuous_gen as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class HallwayConfig(config.InstanceConfig):
    def __init__(self):
        params = {}
        params['x_desc'] = (-5,5,20)
        params['a_desc'] = (-1,1,3)
        
        center = 0
        radius = 0.1
        params['cost_obj'] = mdp.BallSetFn(center,radius)
        
        params['discount'] = 0.99
        self.params = params

    def configure_instance_builder(self):
        gen_fn = gen.HallwayGenerator(**self.params)
        return gen_fn.generate()
