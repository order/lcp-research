"""
Configuration classes are basically there to be small files that
store the data to instantiate a particular solver or instances building
object.
"""

class SolverConfig(object):
    """
    An object that configures and builds a solver generator
    """
    def configure_solver_generator(self):
        """
        Should return a 'Generator'; i.e. something 
        that uses
        """
        raise NotImplementedError()

class InstanceConfig(object):
    """
    An object that configures and builds a builder
    """
    def configure_instance_builder(self):
        """
        Should return a 'builder'; i.e. something 
        like MDPBuilder or
        """
        raise NotImplementedError()
