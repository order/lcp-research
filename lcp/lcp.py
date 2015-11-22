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

    def to_csv(self,filename):
        FH = open(filename,'w')
        for r in self.M.todense():
            FH.write(','.join(map(str,r.tolist()[0])) + '\n')
        FH.write(','.join(map(str,self.q.tolist()[0])) + '\n')
        FH.close()
        

    def __str__(self):
        return '<{0} in R^{1}>'.format(self.name, self.dim)
                
                
            
class MDPLCPObj(LCPObj):
    """LCP generated from an MDP
    """
    def __init__(self,MDP):
        self.MDP = MDP
        (M,q) = MDP.tolcp()
        super(MDPLCPObj,self).__init__(M,q)
        
    def split_vector(self,x):        
        N = self.dim # M is N-by-N
        n = self.MDP.num_states
        A = self.MDP.num_actions
        assert(N == n*(A+1)) # Basic sanity
        
        assert((N,) == x.shape) # x is N-by-1
        
        blocks = [x[(i*n):((i+1)*n)] for i in xrange(A+1)] # break in chucks 
        return [blocks[0],blocks[1:]] # first block is value, rest is flow
           
class ProjectiveLCPObj(LCPObj):
    """An object for a projective LCP where M = Phi U + Pi_bot
    LCPs of this class can be rapidly solved using fast interior point algorithms like the one in "Fast solutions to projective monotone linear complementarity problems" (http://arxiv.org/pdf/1212.6958v1.pdf)  
    """
    def __init__(self,Phi,U,q,**kwargs):
        self.Phi = Phi
        self.U = U
        self.q = q
        self.dim = q.size
        
        assert(util.isvector(q))
        assert(Phi.shape[0] == self.dim)
        assert(U.shape[1] == self.dim)
        
        self.name = kwargs.get('name','Unnamed')

    def to_csv(self,fileroot):
        FH = open(fileroot + '_phi.csv','w')
        for r in self.Phi.todense():
            FH.write(','.join(map(str,r.tolist()[0])) + '\n')
        FH.close()
        FH = open(fileroot + '_u.csv','w')        
        for r in self.U.todense():
            FH.write(','.join(map(str,r.tolist()[0])) + '\n')           
        FH.close()
        FH = open(fileroot + '_q.csv','w')
        FH.write(','.join(map(str,self.q)) + '\n')
        FH.close()
 
        
    def __str__(self):
        return '<{0} in R^{1}'.\
            format(self.name, self.dim)
    
    
    
    
