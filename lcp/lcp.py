import util
import numpy as np

class LCPObj(object):
    """An object that wraps around the matrix M and vector q
    for an LCP
    """
    def __init__(self,M,q,**kwargs):
        self.M = M
        self.q = q
        self.dim = q.size

        assert(len(q.shape) == 1)
        assert(len(M.shape) == 2)
        assert(M.shape[0] == self.dim)
        assert(M.shape[1] == self.dim)
        
        self.name = kwargs.get('name','Unnamed')        

    def __str__(self):
        return '<{0} in R^{1}>'.format(self.name, self.dim)                

# Ever used?
class MDPLCPObj(LCPObj):
    """LCP generated from an MDP
    """
    def __init__(self,MDP):
        self.MDP = MDP
        (M,q) = MDP.build_lcp()
        super(MDPLCPObj,self).__init__(M,q)
        
    def split_vector(self,x):        
        N = self.dim # M is N-by-N
        n = self.MDP.num_states
        A = self.MDP.num_actions
        assert(N == n*(A+1)) # Basic sanity
        
        assert((N,) == x.shape) # x is N-by-1
        
        blocks = [x[(i*n):((i+1)*n)] for i in xrange(A+1)]
        # break in chucks

        # first block is value, rest is flow
        return [blocks[0],blocks[1:]]
           
class ProjectiveLCPObj(LCPObj):
    """An object for a projective LCP where M = Phi U + Pi_bot
    LCPs of this class can be rapidly solved using fast interior 
    point algorithms like the one in "Fast solutions to projective 
    monotone linear complementarity problems" 
    (http://arxiv.org/pdf/1212.6958v1.pdf)  
    """
    def __init__(self,Phi,PtPU,q,**kwargs):
        self.Phi = Phi
        self.PtPU = PtPU # <P,PU>
        self.q = q

        # Shape checking
        (N,) = q.shape
        assert(N == Phi.shape[0])
        (_,K) = Phi.shape
        assert((K,N) == PtPU.shape)
        
        self.name = kwargs.get('name','Unnamed')

    def update_q(self,new_q):
        self.q = new_q
        
    def __str__(self):
        return '<{0} in R^{1}'.\
            format(self.name, self.dim)
    
    
    
    
