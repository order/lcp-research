from solvers import IPIterator,LCPIterator,potential
import numpy as np
import scipy.sparse as sps

from collections import defaultdict

##############################
# Kojima lcp_obj
class KojimaIPIterator(IPIterator,LCPIterator):
    def __init__(self,lcp_obj,**kwargs):
        self.lcp = lcp_obj
        self.M = lcp_obj.M
        self.q = lcp_obj.q
         
        self.centering_coeff = kwargs.get('centering_coeff',0.99)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',
                                             0.99)
        self.central_path_backoff = kwargs.get('central_path_backoff',0.9995)
        
        self.iteration = 0
        
        n = lcp_obj.dim
        
        self.x = kwargs.get('x0',np.ones(n))
        self.y = kwargs.get('y0',np.ones(n))
        self.step = np.nan
        self.del_x = np.full(n,np.nan)
        self.del_y = np.full(n,np.nan)

        self.newton_system = sps.eye(1)
        
        assert(not np.any(self.x < 0))
        assert(not np.any(self.y < 0))

        self.data = defaultdict(list)
        
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
        self.data['x'].append(x)

        (N,) = x.shape

        S = x+y
        print 'min(S):',np.min(S)
        print 'max(S):',np.max(S)
        print 'log ratio:', np.log10(np.max(S) / np.min(S))
        print 'argmin(S)',np.argmin(S)
        print 'argmax(S)',np.argmax(S)

        (P,ip_term,x_term,y_term) = potential(x,y,0.25*N)
        self.data['P'].append(P)
        #self.data['ip term'].append(ip_term)
        #self.data['x term'].append(x_term)
        #self.data['y term'].append(y_term)

        #self.data['min(S)'].append(np.min(S))
        #self.data['max(S)'].append(np.max(S))
        #self.data['ratio(S)'].append(np.log10(np.max(S) / np.min(S)))

        sigma = self.centering_coeff
        beta = self.linesearch_backoff
        backoff = self.central_path_backoff
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)

        self.data['res_norm'].append(np.linalg.norm(r))
        self.data['ip'].append(dot)
            
            # Set up Newton direction equations
        A = sps.bmat([[sps.spdiags(y,0,n,n),sps.spdiags(x,0,n,n)],\
            [-M,sps.eye(n)]],format='csc')          
        b = np.concatenate([sigma * dot / float(n) * np.ones(n) - x*y, r])

        self.newton_system = A
        
        Del = sps.linalg.spsolve(A,b)
        del_x = Del[:n]
        del_y = Del[n:]

        print '||dx||:',np.linalg.norm(del_x)
        print '||dy||:',np.linalg.norm(del_y)
            
        # Get step length so that x,y are strictly feasible after move
        steplen = max(np.max(-del_x/x),
                      np.max(-del_y/y))
        if steplen <= 0:
            steplen = float('inf')
        else:
            steplen = 1.0 / steplen
        steplen = min(1.0, 0.666*steplen + (backoff - 0.666) * steplen**2)
        print 'Steplen', steplen

        #self.data['steplen'].append(steplen)

        # Sigma is beta in Geoff's code
        if(1.0 >= steplen > 0.95):
            sigma *= 0.9 # Long step
        elif(0.1 >= steplen > 1e-3):
            sigma *= 1.2
            sigma = min(sigma,0.99)
        elif (steplen <= 1e-3):
            sigma = 0.99 # Short step
        print 'sigma:',sigma

        #self.data['sigma'].append(sigma)


            # Update point and fields
        self.steplen = steplen
        self.centering_coeff = sigma
              
        self.x = x + steplen * del_x
        self.y = y + steplen * del_y
        self.dir_x = del_x
        self.dir_y = del_y
        self.iteration += 1
        
    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
    def get_step_len(self):
        return self.step
    def get_primal_dir(self):
        return self.del_x
    def get_dual_dir(self):
        return self.del_y
    def get_newton_system(self):
        return self.newton_system
