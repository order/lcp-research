import config
import utils
import mdp.discretizer

class To2DFramesProcessor(config.Processor):
    def process(self,data,**kwargs):
        """
        Using information from the instance builder,
        split the (I x N) data into (I x A x X x Y) frames.

        The instance builder should be a discretizer
        """

        (I,N) = data.shape
        
        assert('instance_builder' in kwargs)
        inst_builder = kwargs['instance_builder']
        assert(issubclass(type(inst_builder),
                          mdp.discretizer.MDPDiscretizer))
        A = inst_builder.get_num_actions()
        n = inst_builder.get_num_nodes()
        assert(N == (A+1)*n)
 
        (x,y) = inst_builder.get_basic_lengths()
        assert(n >= x*y)
        
        return utils.processing.split_into_frames(data,A,n,x,y)
