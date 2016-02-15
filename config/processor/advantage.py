import config
import utils
import mdp.discretizer
import numpy as np

class AdvantageFrameProcessor(config.Processor):
    def process(self,data,**kwargs):
        """
        Takes as input a MDP frame tensor (I,A,X,Y)
        Returns the flow difference between 
        """
        if 2 == len(data.shape):
            print 'Run To2DFramesProcessor first' 
        (I,A,X,Y) = data.shape
        assert(A >= 2)
        
        SortedFlowFrames = np.sort(data[:,1:,:,:],axis=1)
        Adv = SortedFlowFrames[:,-1,:,:] - SortedFlowFrames[:,-2,:,:]
        assert(not np.any(Adv < 0))
        return Adv
