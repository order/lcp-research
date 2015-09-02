import numpy as np
import math
from mdp.acrobot import AcrobotRemapper
from mdp.simulator import ChainSimulator

def simulate_test():
    physics = AcrobotRemapper(l1=2)
    init_state = np.array([[math.pi/4.0,0.0,0,0]])

    physics.remap(init_state,action=0)
    physics.forward_kinematics(init_state)

    dim = 2
    sim = ChainSimulator(dim,physics)
    sim.simulate(init_state,1000)
    
def mdp_test():
    theta_lim = 2*math.pi
    theta_n = 35
    
    a_lim = 1
    a_n = 3    
    
    basic_mapper = InterpolatedGridNodeMapper(np.linspace(-x_lim,x_lim,x_n),np.linspace(-v_lim,v_lim,v_n))

    