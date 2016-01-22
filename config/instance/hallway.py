import numpy as np
import config
import mdp.hallway as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class StubGenerator(object):
    """
    Stupid function
    """
    def __init__(self,generator):
        self.generator
    def generate(self):
        return self.generator

class SmallDoubleIntegratorConfig(config.Config):
    def __init__(self):
        params = {}
        params['wheel_slip'] = 0.1
        params['num_states'] = 25
                
        params['discount'] = 0.99
        self.params = params

    def build(self):
        return StubGenerator(gen.HallwayGenerator(**self.params))
