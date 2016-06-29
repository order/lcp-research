import numpy as np
from solvers import IPIterator

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
        assert(isinstance(iteration, IPIterator))
        self.data.append(iteration.get_primal_dir())

class DualDirRecorder(Recorder):
    def report(self,iteration):
        assert(isinstance(iteration, IPIterator))
        self.data.append(iteration.get_dual_dir())

class ValueRecorder(Recorder):
    def report(self,iteration):
        self.data.append(iteration.get_value_vector())

class StepLenRecorder(Recorder):
    def report(self,iteration):
        assert(isinstance(iteration, IPIterator))
        self.data.append(iteration.get_step_len())

class NewtonCondNumRecorder(Recorder):
    def report(self,iteration):
        assert(isinstance(iteration, IPIterator))
        A = iteration.get_newton_system()
        try:
            A = A.todense()
        except:
            pass
        assert(2 == len(A.shape))
        self.data.append(np.linalg.cond(A))
