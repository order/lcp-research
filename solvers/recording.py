import numpy as np

############################
# Recording functions


class Recorder(object):
    def __init__(self):
        self.reset()
    def report(self,iteration):
        raise NotImplementedError()
    def reset(self):
        self.data = []
        
class PrimalRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_primal_vector())

class DualRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_dual_vector())

class PrimalDirRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_primal_dir())

class DualDirRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_dual_dir())

class ValueRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_value_vector())

class StepLenRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_step_len())


