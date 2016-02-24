import config
import utils
import mdp.discretizer
import numpy as np

class ValueFramesProcessor(config.Processor):
    def process(self,data,**kwargs):
        """
        Using information from the instance builder,
        split the (I,N) value data into (I,X1,X2,...) frames.

        The instance builder should be a discretizer
        """

        (I,N) = data.shape
        
        assert('instance_builder' in kwargs)
        inst_builder = kwargs['instance_builder']
        assert(issubclass(type(inst_builder),
                          mdp.discretizer.MDPDiscretizer))
        n = inst_builder.get_num_nodes()
        assert(N == n)
 
        lengs = inst_builder.get_basic_lengths()
        assert(n >= np.prod(lengs))

        frames = utils.processing.split_into_frames(data,0,n,lengs).squeeze()
        
        return frames
