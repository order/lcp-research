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
