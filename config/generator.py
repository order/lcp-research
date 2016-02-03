import numpy as np

class Generator(object):
    def generate(self,**kwargs):
        """
        Generate something
        """
        raise NotImplementedError()    

class SolverGenerator(Generator):
    def generate(self,discretizer):
        raise NotImplementedError()
    def extract(self,solver):
        raise NotImplementedError()

class StubGenerator(Generator):
    """
    Way of wrapping a Builder so it is a Generator
    A Builder is something that builds an object (e.g. Discretizer)
    A Generator is something that configures and generates a Builder
    """
    def __init__(self,builder):
        self.builder = builder
    def generate(self):
        return self.builder

def add_trn(gen,solver):
    """
    Add termination, recording, and notification
    """
    # Add termination conditions
    solver.termination_conditions.extend(
        gen.termination_conditions.values())

    # Set up recorders
    gen.recorder_names = gen.recorders.keys()
    solver.recorders.extend(gen.recorders.values())
    
    # Set up notification
    solver.notifications.extend(gen.notifications.values())

def basic_extract(gen,solver):
    # Extract the value information
    # TODO: generalize
    names = gen.recorder_names
    assert(len(names) == len(solver.recorders))
    
    data = {}
    for i in xrange(len(names)):
        data[names[i]] = np.array(solver.recorders[i].data)

    return data

