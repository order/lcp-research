import numpy as np

from solvers import LCPIterator
from lcp import LCPObj

class ExtraGradientIterator(LCPIterator):
    def __init__(self,lcp,**kwargs):
        self.lcp = lcp
        self.tau = kwargs.get('tau',1e-6)
        
        N = lcp.q.size
        self.x  = kwargs.get('x0',np.ones(N))
        
        self.iteration = 0

        self.data = defaultdict(list)

        
    def next_iteration(self): 
        N = self.lcp.q.size

        x = self.x
        t = self.tau
        F = self.lcp.F
        
        z = np.maximum(0,x - t*F(x)) # Gradient
        self.x = np.maximum(0,x - t*F(z)) # Extra gradient
        self.iteration += 1 

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.lcp.F(self.x)
    def get_iteration(self):
        return self.iteration
