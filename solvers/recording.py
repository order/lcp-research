import numpy as np

############################
# Recording functions


class Recorder(object):
    def report(self,iteration):
        raise NotImplementedError()
    def reset(self):
        raise NotImplementedError()
        
class PrimalRecorder(Recorder):
    def __init__(self):
        self.reset()
        
    def report(self,iteration):
        self.data.append(iteration.get_primal_vector())

    def reset(self):
        self.data = []
   
