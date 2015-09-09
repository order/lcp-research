import numpy as np
import math
from mdp.bicycle import BicycleRemapper
from mdp.simulator import BicycleSimulator
from mdp.state_remapper import AngleWrapStateRemaper,RangeThreshStateRemapper
from mdp.node_mapper import InterpolatedGridNodeMapper
from mdp.discretizer import ContinuousMDPDiscretizer
from mdp.costs import QuadraticCost

import lcp.solvers

def simulate_test():
    physics = BicycleRemapper()
    S = len(physics.dim_ids)
    dim_ids = physics.dim_ids
    
    state = np.array(np.zeros((1,S)))
    state[0,dim_ids['omega']] = 0.01
    state[0,dim_ids['xf']] = physics.params.l
    state[0,dim_ids['psi']] = -np.pi / 2.0

    action = np.zeros(2)
    
    sim = BicycleSimulator(physics)
    sim.simulate(state,action,25)
    
simulate_test()