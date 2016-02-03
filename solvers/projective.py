from solvers import LCPIterator,IPIterator

import numpy as np
import scipy as sp
import scipy.sparse as sps

class ProjectiveIPIterator(LCPIterator,IPIterator):
    def __init__(self,proj_lcp_obj,**kwargs):
        self.centering_coeff = kwargs.get('centering_coeff',0.99)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',0.99)
        self.central_path_backoff = kwargs.get('central_path_backoff',0.9995)
        
        self.proj_lcp_obj = proj_lcp_obj
        self.q = proj_lcp_obj.q
        Phi = proj_lcp_obj.Phi
        U = proj_lcp_obj.U
        self.Phi = Phi
        self.U = U
        (N,k) = Phi.shape
         
        # Preprocess
        PtP = (Phi.T).dot(Phi) # FTF in Geoff's code
        assert((k,k) == PtP.shape)
        self.PtPUP = (PtP.dot(U)).dot(Phi)
        assert((k,k) == self.PtPUP.shape)        
        self.PtPU_P = (PtP.dot(U) - Phi.T)
        # FMI in Geoff's code; I sometimes call it Psi in my notes
        assert((k,N) == self.PtPU_P.shape)            
        self.PtP = PtP

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(k))
        self.steplen = np.nan
        self.dir_x = np.full(N,np.nan)
        self.dir_y = np.full(N,np.nan)
        self.dir_w = np.full(k,np.nan)
        
        assert((N,) == self.x.shape)
        assert((N,) == self.y.shape)
        assert((k,) == self.w.shape)

        self.iteration = 0
        
    def next_iteration(self):
        """
        A projective interior point algorithm based on "Fast solutions to projective monotone LCPs" by Geoff Gordon. 
        If M = \Phi U + \Pi_\bot where Phi is low rank (k << n) and \Pi_\bot is projection onto the nullspace of \Phi^\top, 
        then each iteration only costs O(nk^2), rather than O(n^{2 + \eps}) for a full system solve.

        This is a straight-forward implementation of Figure 2
        """

        sigma = self.centering_coeff
        beta = self.linesearch_backoff
        backoff = self.central_path_backoff
     
        Phi = self.Phi
        U = self.U
        q = self.q
        
        assert(len(Phi.shape) == 2)
        (N,k) = Phi.shape
        assert((k,N) == U.shape)
        assert((N,) == q.shape)

        PtP = self.PtP # FTF in Geoff's code
        PtPUP = self.PtPUP
        PtPU_P = self.PtPU_P 

        x = self.x
        y = self.y
        w = self.w
        assert((N,) == x.shape)
        assert((N,) == y.shape)
        assert((k,) == w.shape)

        # Step 3: form right-hand sides
        g = sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        p = x - y + q - Phi.dot(w)
        assert((N,) == g.shape)
        assert((N,) == p.shape)
        
        # Step 4: Form the reduced system G dw = h
        X = sps.diags(x,0)
        Y = sps.diags(y,0)
        Z = sps.lil_matrix((N,k))
        I = sps.eye(N)
        M = Phi.dot(U)
        
        if False:
            # Un-reduced system
            NewtonSystem = sps.bmat([[Y, X ,Z],\
                [-M,I,Z],\
                [-I,I,Phi]])
            print 'NewtonSystem: '
            print '\tShape:',NewtonSystem.shape
            cond = np.linalg.cond(NewtonSystem.todense())
            print '\tCond: {0:.3g}'.format(cond)         
            # Eliminate the x block
            ReductedNewtonSystem = sps.bmat([[X+Y,Y.dot(Phi)],\
                [I - M, -M.dot(Phi)]])
                
            print 'ReductedNewtonSystem: '
            print '\tShape:',ReductedNewtonSystem.shape
            cond = np.linalg.cond(ReductedNewtonSystem.todense())
            print '\tCond: {0:.3g}'.format(cond)      
            
        # Eliminate both the x and y blocks
        inv_XY = sps.diags(1.0/(x + y),0)
        A = (PtPU_P.dot(inv_XY))
        assert((k,N) == A.shape)        
        G = ((A.dot(Y)).dot(Phi) - PtPUP)

        G_sparse = float(G.nnz) / float(np.prod(G.shape))
        assert((k,k) == G.shape)
        
        h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
        assert((k,) == h.shape)
            
        # Step 5: Solve for del_w
        # G is essentially dense
        if G_sparse > 0.01:
            del_w = np.linalg.lstsq(G.toarray(),h)[0]
        else:
            del_w = sps.linalg.spsolve(G,h)
        assert((k,) == del_w.shape)
            
            # Step 6
        Phidw= Phi.dot(del_w)
        assert((N,) == Phidw.shape)
        del_y = inv_XY.dot(g + y*p - y*Phidw)
        assert((N,) == del_y.shape)
        
        # Step 7
        del_x = del_y+Phidw-p
        assert((N,) == del_x.shape)
        
            
        # Step 8 Step length
        steplen = max(np.amax(-del_x/x),np.amax(-del_y/y))
        if steplen <= 0:
            steplen = float('inf')
        else:
            steplen = 1.0 / steplen
        steplen = min(1.0, 0.666*steplen + (backoff - 0.666) * steplen**2)
            
        # Sigma is beta in Geoff's code
        if(steplen > 0.95):
            sigma = 0.05 # Long step
        else:
            sigma = 0.5 # Short step

        # Update point and fields
        self.steplen = steplen
        self.centering_coeff = sigma
              
        self.x = x + steplen * del_x
        self.y = y + steplen * del_y
        self.w = w + steplen * del_w
        self.dir_x = del_x
        self.dir_y = del_y
        self.dir_w = del_w
        self.iteration += 1

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
    def get_step_len(self):
        return self.steplen
