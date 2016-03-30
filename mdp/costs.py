import state_functions

class CostWrapper(object):
    def __init__(self,state_fn):
        self.state_fn = state_fn
    def cost(self,states,action):
        return self.state_fn.evaluate(states)
