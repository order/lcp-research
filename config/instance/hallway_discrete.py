import numpy as np
import config
import mdp.hallway as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class HallwayConfig(config.InstanceConfig):
    def __init__(self):
        params = {}
        params['wheel_slip'] = 0.1
        params['num_states'] = 25
                
        params['discount'] = 0.99
        self.params = params

    def configure_instance_builder(self):
        return gen.HallwayBuilder(**self.params)
