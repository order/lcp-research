class TransitionFunction(object):
    """
    Abstract class defining (state,action)-to-state mapping. 

    Transition functions may be stochastic; setting number of samples
    should yield an independent redraw
    """
    def transition(self,states,action,samples=1):
        """
        All state remappers must implement this.
        """
        raise NotImplementedError()
