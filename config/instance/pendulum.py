import numpy as np
import config
import config.instance.gens.pendulum_gen as gen
import mdp

"""
Weird issue: don't use a "from x import y"
"""

class PendulumConfig(config.InstanceConfig):
    def __init__(self):
        params = {}
        params['q_n'] = 20
        params['dq_desc'] = (-6,6,20)
        params['a_desc'] = (-1,1,3)
        
        center = np.zeros(2)
        radius = 0.25
        nudge = np.pi /16.0
        cost_obj = mdp.TargetZoneFn(np.array([\
                                            [np.pi-nudge,np.pi+nudge],\
                                            [-nudge,nudge]]))        
        params['cost_obj'] = cost_obj
        
        params['discount'] = 0.99
        self.params = params

    def configure_instance_builder(self):
        gen_fn = gen.PendulumGenerator(**self.params)
        return gen_fn.generate()
