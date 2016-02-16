import numpy as np

class ProblemGenerator(object):
    def generate_problem(self):
        raise NotImplementedError()

class DiscretizerGenerator(object):
    """
    Generates a discretizer from a problem
    """
    def generate_discretizer(self,problem):
        raise NotImplementedError()        

class SolverGenerator(object):
    def generate_solver(self,builder):
        """
        Builder must be able to create whatever objects
        generator needs
        """
        raise NotImplementedError()
    def extract(self,solver):
        raise NotImplementedError()

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

