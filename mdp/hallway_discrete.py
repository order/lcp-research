import numpy as np
import mdp
import scipy.sparse as sps
from utils.parsers import KwargParser

class HallwayBuilder(mdp.MDPBuilder):
    """
    This is a function that 
    """
    def __init__(self,**kwargs):
        parser = KwargParser()
        parser.add('wheel_slip')
        parser.add('num_states')
        parser.add('discount')
        args = parser.parse(kwargs)

        self.__dict__.update(args)
        assert(0 < self.discount < 1)
        assert(0 <= self.wheel_slip <= 1)
        
    def build_mdp(self):
        N = self.num_states
        w = self.wheel_slip
        actions = [-1,1]

        # Gaussian costs
        cost = 1.0  - np.exp(-np.power(np.linspace(-2,2,N),2.0))
        costs = [cost,cost] # Same for both actions

        transitions = []
        for i in actions:
            moved = sps.diags(np.ones(N-1),i,format='lil')
            stay = sps.eye(N,format='lil')
            P = w * stay + (1 - w) * moved

            # Wrap around
            # if i = -1 then (1 + i) / 2 = 0; is 1 if i = 1
            P[(N-1)*(1+i)/2,(N-1)*(1-i)/2] =  (1 - w)
            transitions.append(P)       
        weights = np.ones(N)
            
        return mdp.DiscreteMDP(transitions,
                               costs,
                               actions,
                               self.discount,
                               weights,
                               name='Hallway')

    def get_node_states(self):
        return np.full((self.num_states,1),np.nan)
