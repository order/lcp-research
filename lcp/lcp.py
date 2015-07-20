from util import *

class LCP(object):
    """An object that wraps around the matrix M and vector q
    for an LCP
    """
    def __init__(self,M,q):
        self.M = M
        self.q = q
        self.dim = q.size
        assert(isvector(q))
        assert(issquare(M))
        assert(M.shape[0] == self.dim)
        
        self.name = 'Unnamed'

    def __str__(self):
        return '<{0} LCP in R^{1}'.\
            format(self.name, self.dim)
