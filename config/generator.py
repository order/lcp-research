class Generator(object):
    def generate(self,**kwargs):
        """
        Generate something
        """
        raise NotImplementedError()    

class SolverGenerator(Generator):
    def extract(self):
        """
        Extract data from the solver
        """
        raise NotImplementedError


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

