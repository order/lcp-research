from lcp import LCPIterator

class ProjectiveIteration(LCPIterator):
    def __init__(self,proj_lcp_obj,**kwargs):
        self.centering_coeff = kwargs.get('centering_coeff',0.99)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',0.99)
        self.central_path_backoff = kwargs.get('central_path_backoff',0.9995)
        
        self.proj_lcp_obj = proj_lcp_obj
        self.q = proj_lcp_obj.q
        self.Phi = proj_lcp_obj.Phi
        self.U = proj_lcp_obj.U
        
        # We expect Phi and U to be both either dense, or both sparse (no mixing)
        # If sparse, we want them to be csr or csc (these can be mixed)
        # self.linsolve is set based on whether we are in the sparse regime or the dense one.
        # TODO: also allow mixed sparseness (e.g. Phi dense, U csr)
        if isinstance(self.Phi,np.ndarray):
            assert(isinstance(self.U,np.ndarray))
            self.linsolve = lambda A,b: np.linalg.lstsq(G,h)[0]
        else:
            assert(isinstance(self.Phi,sps.csc_matrix) or isinstance(self.Phi,sps.csr_matrix))
            assert(isinstance(self.U,sps.csc_matrix) or isinstance(self.U,sps.csr_matrix))
            self.linsolve = lambda A,b: sps.linalg.lsqr(G,h)[0]
         
        # Preprocess
        self.PtP = (Phi.T).dot(Phi) # FTF in Geoff's code
        self.PtPUP = (PtP.dot(U)).dot(Phi)
        self.PtPU_P = (PtP.dot(U) - Phi.T) # FMI in Geoff's code; I sometimes call it Psi in my notes
        
        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(k))

        self.iteration = 0
        
    def next_iteration(self):
        """
        A projective interior point algorithm based on "Fast solutions to projective monotone LCPs" by Geoff Gordon. 
        If M = \Phi U + \Pi_\bot where Phi is low rank (k << n) and \Pi_\bot is projection onto the nullspace of \Phi^\top, then each iteration only costs O(nk^2), rather than O(n^{2 + \eps}) for a full system solve.

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
        assert(ismatrix(U,k,N))
        assert(isvector(q,N))

        PtP = self.PtP # FTF in Geoff's code
        assert(issquare(PtP,k))
        PtPUP = self.PtPUP
        assert(issquare(PtPUP,k))
        PtPU_P = self.PtPU_P 
        assert(ismatrix(PtPU_P,k,N))

        x = self.x
        y = self.y
        w = self.w
        assert(isvector(x,N))
        assert(isvector(y,N))
        assert(isvector(w,k))

        # Step 3: form right-hand sides
        g = sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        p = x - y + q - Phi.dot(w)
        assert(isvector(g,N))
        assert(isvector(p,N))
        
        # Step 4: Form the reduced system G dw = h
        inv_XY = 1.0/(x + y)
        A = PtPU_P * inv_XY
        assert(ismatrix(A,k,N))    
        G = (A * y).dot(Phi) - PtPUP
        assert(issquare(G,k))
        h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
        assert(isvector(h,k))
            
        # Step 5: Solve for del_w
        del_w = self.linsolve(G,h) # Uses either dense or sparse least-squares
        assert(isvector(del_w))
        assert(isvector(del_w,k))
            
            # Step 6
        Phidw= Phi.dot(del_w)
        assert(isvector(Phidw,N))
        del_y = inv_XY*(g + y*p - y*Phidw)
        assert(isvector(del_y,N))
        
        # Step 7
        del_x = del_y+Phidw-p
        assert(isvector(del_x,N))      
            
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
              
        self.x = x + steplen * del_x
        self.y = y + steplen * del_y
        self.w = w + steplen * del_w   
