import numpy as np

############################
# Recording functions
class Recorder(object):
    def report(self,iteration):
        raise NotImplementedError()
        
class PrimalRecorder(Recorder):
    def __init__(self):
        pass
        
    def report(self,iteration):
        return iteration.get_primal_vector()
   
