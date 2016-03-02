import config
import numpy as np
import utils

class ResidualProcessor(config.Processor):
    def process(self,data,**kwargs):
        """
        Convert the (I,N) frames into (I,N)
        movie that encodes the Bellman residual

        The instance builder should be a discretizer
        """

        (I,N) = data.shape
        mdp_obj = kwargs['objects']['mdp']
        n = mdp_obj.num_states
        
        residual = np.zeros((I,N))
        for i in xrange(I):
            residual[i,:n] = mdp_obj.get_value_residual(data[i,:n])
            
        return residual

