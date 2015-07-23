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
    
    def get_qr(self):
        """Get QR factorization; factorization cached
        """
        if not hasattr(self,'Q'):
            self.Q, self.R = scipy.linalg.qr(self.Phi,mode='economic')
            self.rank = self.Q.shape[1]
            # Q is n x k
            # R is k x n
            assert(self.Q.shape[0] == self.dim)
            assert(self.R.shape[1] == self.dim)
            assert(self.R.shape[0] == self.rank)
            
        return (self.Q,self.R)        

    def __str__(self):
        return '<{0} Projective LCP in R^{1}'.\
            format(self.name, self.dim)

def pinv(Q,R,x,**kwargs):
    """ Calculate the y s.t. Ay is close to x via the pseudo inverse of A = QR
    """
    # b = A^T x
    b = np.dot(R.T,np.dot(Q.T,x))
    # (R.T*R)y = b
    y = scipy.linalg.solve(np.dot(R.T,R),b)
    return y    
 
def pi_project(Phi,x,**kwargs):
    """ Project x using the pseudo inverse of Phi:
    Pi x = Phi (Phi.T Phi)^-1 Phi.T x
    """
    if 'Q' in kwargs and 'R' in kwargs:
        Q = kwargs['Q']
        R = kwargs['R']
    else:
        (Q,R) = scipy.linalg.qr(A)
    y = pinv(Q,R,x)
    return Q.dot(R.dot(y))
    
    
    
    