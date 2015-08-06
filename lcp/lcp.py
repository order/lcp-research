import util
import numpy as np

class LCPObj(object):
    """An object that wraps around the matrix M and vector q
    for an LCP
    """
    def __init__(self,M,q):
        self.M = M
        self.q = q
        self.dim = q.size
        assert(util.isvector(q))
        assert(util.issquare(M))
        assert(M.shape[0] == self.dim)
        
        self.name = 'Unnamed'

    def __str__(self):
        return '<{0} LCP in R^{1}'.\
            format(self.name, self.dim)
            
    def write_to_csv(self,filename):
        FH = open(filename,'w')
        D = np.array(self.M.todense())
        for row in D:
            FH.write(','.join(map(str,row.flatten())) + '\n')
        FH.write(','.join(map(str,self.q)))
                
                
            
class MDPLCPObj(LCPObj):
    """LCP generated from an MDP
    """
    def __init__(self,MDP):
        self.MDP = MDP
        (M,q) = MDP.tolcp()
        super(MDPLCPObj,self).__init__(M,q)
            
class ProjectiveLCPObj(LCPObj):
    """An object for a projective LCP where M = Phi U + Pi_bot
    LCPs of this class can be rapidly solved using fast interior point algorithms like the one in "Fast solutions to projective monotone linear complementarity problems" (http://arxiv.org/pdf/1212.6958v1.pdf)  
    """
    def __init__(self,Phi,U,q):
        self.Phi = Phi
        self.U = U
        self.q = q
        self.dim = q.size
        
        assert(util.isvector(q))
        assert(Phi.shape[0] == self.dim)
        assert(U.shape[1] == self.dim)
        
        self.name = 'Unnamed'

    def __str__(self):
        return '<{0} Projective LCP in R^{1}'.\
            format(self.name, self.dim)

    def write_to_csv(self,filename):
        FH = open(filename,'w')
        if hasattr(self.Phi,'todense'):
            DP = np.array(self.Phi.todense())
            DU = np.array(self.U.todense())
        else:
            DP = self.Phi
            DU = self.U

        for row in DP:
            FH.write(','.join(map(str,row.flatten())) + '\n')
        for row in DU:
            FH.write(','.join(map(str,row.flatten())) + '\n')
        FH.write(','.join(map(str,self.q)))
    
    
    
    
