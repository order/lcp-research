import config
import numpy as np

class PolicyProcessor(config.Processor):
    def process(self,data,**kwargs):
        """
        Convert the (I,A,X,Y) frames into (I,X,Y)
        movie that encodes the policies

        The instance builder should be a discretizer
        """

        assert(4 == len(data.shape))
        (I,A,X,Y) = data.shape
        policy = np.argmax(data[:,1:,:,:],axis=1)
        assert((I,X,Y) == policy.shape)
        return policy

