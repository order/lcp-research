from solvers import IPIterator,LCPIterator,potential,steplen_heuristic
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
                 
        self.iteration = 0
        
        n = lcp_obj.dim
        
        self.x = kwargs.get('x0',np.ones(n))
        self.y = kwargs.get('y0',np.ones(n))

        self.sigma = 0.95
        self.steplen = np.nan

        self.dir_x = np.full(n,np.nan)
        self.dir_y = np.full(n,np.nan)
        
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
        self.data['y'].append(y)

        (N,) = x.shape

        S = x+y
        print 'min(S):',np.min(S)
        print 'max(S):',np.max(S)
        print 'log ratio:', np.log10(np.max(S) / np.min(S))
        print 'argmin(S)',np.argmin(S)
        print 'argmax(S)',np.argmax(S)

        (P,ip_term,x_term,y_term) = potential(x,y,np.sqrt(N))
        #self.data['P'].append(P)

        sigma = self.sigma
        
        r = (M.dot(x) + q) - y 
        dot = x.dot(y)

        #self.data['res_norm'].append(np.linalg.norm(r))
        self.data['ip'].append(dot / float(N))
            
        # Set up Newton direction equations
        X = sps.diags(x,0)
        Y = sps.diags(y,0)
        I = sps.eye(n)
        A = sps.bmat([[Y,X],
                      [-M,I]],format='csc')          
        b = np.concatenate([sigma * dot / float(n) * np.ones(n) - x*y, r])
        
        Del = sps.linalg.spsolve(A,b)
        dir_x = Del[:n]
        dir_y = Del[n:]

        print '||dx||:',np.linalg.norm(dir_x)
        print '||dy||:',np.linalg.norm(dir_y)
        self.data['dx'].append(dir_x)
        self.data['dy'].append(dir_y)

        steplen = steplen_heuristic(x,dir_x,y,dir_y,0.6)
        #steplen = min(0.1,steplen)
        print 'Steplen', steplen

        self.data['steplen'].append(steplen)

        if(1.0 >= steplen > 0.95):
            sigma *= 0.95  # Long step
        elif(0.1 >= steplen > 1e-3):
            sigma = 0.5 + 0.5*sigma
        elif (steplen <= 1e-3):
            sigma = 0.9 + 0.1*sigma
        #sigma = 0.9
        sigma = min(0.999,max(0.1,sigma))
        print 'sigma:',sigma
        self.data['sigma'].append(sigma)

        # Update point and fields
        self.steplen = steplen
        self.sigma = sigma
              
        self.x = x + steplen * dir_x
        self.y = y + steplen * dir_y
        self.dir_x = dir_x
        self.dir_y = dir_y
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
        return self.dir_x
    def get_dual_dir(self):
        return self.dir_y
    def get_newton_system(self):
        return self.newton_system
