from solvers import LCPIterator
import numpy as np
import scipy.sparse as sps

##############################
# Kojima lcp_obj
class KojimaIterator(LCPIterator):
    def __init__(self,lcp_obj,**kwargs):
        self.lcp = lcp_obj
        self.M = lcp_obj.M
        self.q = lcp_obj.q
        
        self.centering_coeff = kwargs.get('centering_coeff',0.1)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',0.8)  
        
        self.iteration = 0
        
        n = lcp_obj.dim
        
        self.x = kwargs.get('x0',np.ones(n))
        self.y = kwargs.get('y0',np.ones(n))
        assert(not np.any(self.x < 0))
        assert(not np.any(self.y < 0))
        
    def next_iteration(self):
        """Interior point solver based on Kojima et al's
        "A Unified Approach to Interior Point Algorithms for lcp_objs"
        Uses a log-barrier w/centering parameter
        (How is this different than the basic scheme a la Nocedal and Wright?)
        """
        M = self.lcp.M
        q = self.lcp.q
        n = self.lcp.dim
        
        x = self.x
        y = self.y

        sigma = self.centering_coeff
        beta = self.linesearch_backoff
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)
            
            # Set up Newton direction equations
        A = sps.bmat([[sps.spdiags(y,0,n,n),sps.spdiags(x,0,n,n)],\
            [-M,sps.eye(n)]],format='csc')          
        b = np.concatenate([sigma * dot / float(n) * np.ones(n) - x*y, r])
                
        dir = sps.linalg.spsolve(A,b)
        dir_x = dir[:n]
        dir_y = dir[n:]
            
        # Get step length so that x,y are strictly feasible after move
        step = 1
        x_cand = x + step*dir_x
        y_cand = y + step*dir_y    
        while np.any(x_cand < 0) or np.any(y_cand < 0):
            step *= beta
            x_cand = x + step*dir_x
            y_cand = y + step*dir_y

        x = x_cand
        y = y_cand
        
        self.x = x
        self.y = y
        self.iteration += 1
        
    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
