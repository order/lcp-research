import numpy as np
import config
import config.generator
import mdp.hallway as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class SmallDoubleIntegratorConfig(config.Config):
    def __init__(self):
        params = {}
        params['wheel_slip'] = 0.1
        params['num_states'] = 25
                
        params['discount'] = 0.99
        self.params = params

    def build(self):
        return config.generator.StubGenerator(
            gen.HallwayBuilder(**self.params))
