import numpy as np
import scipy as sp
import scipy.sparse as sps

from solvers import *
from steplen import *

import utils
import matplotlib.pyplot as plt

from collections import defaultdict

import time

def solve_system(A,b):
    # Try to smartly use the right solver for A
    if isinstance(A,sps.spmatrix):
        A.eliminate_zeros()
        fillin = float(A.nnz) / float(A.size)
    else:
        fillin = 1.0
    
    if fillin > 0.1:
        if isinstance(A,sps.spmatrix):
            A = A.toarray()
        x = np.linalg.solve(A,b)
    else:
        x = sps.linalg.spsolve(A,b)
    return x
    
###################################################################
# Start of the iterator
class ProjectiveIPIterator(LCPIterator,IPIterator,BasisIterator):
    def __init__(self,plcp,**kwargs):
        print "Initializing projective solver..."
        self.mdp_obj = kwargs.get('mdp_obj',None)
        self.lcp_obj = kwargs.get('lcp_obj',None)
                
        self.plcp = plcp
        
        Phi = plcp.Phi
        U = plcp.U
        q = plcp.q
        
        (N,K) = Phi.shape
        assert (K,N) == U.shape
        assert (N,) == q.shape

        self.q = plcp.q
        self.Ptq = Phi.T.dot(self.q)
        assert (K,) == self.Ptq.shape

        print "Updating matrices..."
        self.update_P_PtPU(Phi,U)

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',Phi.T.dot(q))

        assert (N,) == self.x.shape
        assert (N,) == self.y.shape
        assert (K,) == self.w.shape
        
        w_res = self.x - self.y + q - Phi.dot(self.w)
        #print 'W residual', np.linalg.norm(w_res)
        #assert(np.linalg.norm(w_res) < 1e-12)

        self.sigma = 0.5
        self.steplen = np.nan
        
        self.dir_x = np.full(N,np.nan)
        self.dir_y = np.full(N,np.nan)
        self.dir_w = np.full(K,np.nan)
        
        assert((N,) == self.x.shape)
        assert((N,) == self.y.shape)
        assert((K,) == self.w.shape)

        self.data = defaultdict(list)

        self.iteration = 0

        self.verbose = True
        
    def next_iteration(self):
        """
        A projective interior point algorithm based on 
        "Fast solutions to projective monotone LCPs" by Geoff Gordon. 
        If M = \Phi U + \Pi_\bot where Phi is low rank (k << n) and 
        \Pi_\bot is projection onto the nullspace of \Phi^\top, 
        then each iteration only costs O(nk^2), rather than O(n^{2 + \eps}) 
        for a full system solve.

        This is an implementation of Figure 2
        """
        iter_start = time.time()
        Phi = self.Phi
        U = self.U
        q = self.q
        
        assert(len(Phi.shape) == 2)
        (N,k) = Phi.shape
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
        self.data['x'].append(x)
        self.data['y'].append(y)
        self.data['w'].append(w)
        

        (P,ip_term,x_term,y_term) = potential(x,y,np.sqrt(N))
        #self.data['P'].append(P)

        dot = x.dot(y)
        if self.verbose:
            print 'IP/N', dot / float(N)
        self.data['ip'].append(dot / float(N))

        # Step 3: form right-hand sides
        sigma = self.sigma
                
        # Step 4: Form the reduced system G dw = h
        newton_start = time.time()
        if self.verbose:
            print 'Forming Netwon system...'
        (G,g,h) =  self.form_Gh(x,y,sparse=True)
        if self.verbose:
            print 'Elapsed time', time.time() - newton_start

        # Step 5: Solve for dir_w
        solver_start = time.time()
        dir_w = solve_system(G,h)
        if self.verbose:
            print 'Solver dw residual', np.linalg.norm(G.dot(dir_w) - h)
        assert((k,) == dir_w.shape)
        if self.verbose:
            print 'Elapsed time', time.time() - solver_start
        
        # Step 6: recover dir_y
        Phidw= Phi.dot(dir_w)
        assert((N,) == Phidw.shape)
        S = x+y
       
        inv_XY = sps.diags(1.0/S,0)
        dir_y = inv_XY.dot(g - y*Phidw)
        assert((N,) == dir_y.shape)
        
        # Step 7: recover dir_x
        dir_x = dir_y + Phidw
        assert((N,) == dir_x.shape)

        if False:
            print '!!! Checking against full Newton system'
            (A,b) = self.form_full_newton(x,y)
            dir = solve_system(A,b)
            dir_x_alt = dir[:N]
            dir_y_alt = dir[N:(2*N)]
            dir_w_alt = dir[(2*N):]

            print '||dx res||:',np.linalg.norm(dir_x - dir_x_alt)
            print '||dy res||:',np.linalg.norm(dir_y - dir_y_alt)
            print '||dw res||:',np.linalg.norm(dir_w - dir_w_alt)            
            
        if self.verbose:
            print '||dx||:',np.linalg.norm(dir_x)
            print '||dy||:',np.linalg.norm(dir_y)
            print '||dw||:',np.linalg.norm(dir_w)
        #self.data['dx'].append(dir_x)
        #self.data['dy'].append(dir_y)
        #self.data['dw'].append(dir_w)
        
        steplen = steplen_heuristic(x,dir_x,y,dir_y,0.6)
        if self.verbose:
            print 'Steplen', steplen
        self.data['steplen'].append(steplen)
        self.steplen = steplen

        sigma = sigma_heuristic(sigma,steplen)
        if self.verbose:
            print 'sigma:',sigma
        self.data['sigma'].append(sigma)
        self.sigma = sigma      
              
        self.x = x + steplen * dir_x
        self.y = y + steplen * dir_y
        self.w = w + steplen * dir_w
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_w = dir_w

        print 

        #w_res = dir_x - dir_y - Phidw
        #print 'W residual:', np.linalg.norm(w_res)
        
        self.iteration += 1
        if self.verbose:
            print 'Total iteration time', time.time() - iter_start
        
    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_value_vector(self):
        n = self.mdp_obj.num_states
        return self.x[:n]
    def get_iteration(self):
        return self.iteration
    def get_step_len(self):
        return self.steplen        

    def update_P_PtPU(self,Phi,U):        
        self.Phi = Phi
        self.U = U
        PtPU = (Phi.T.dot(Phi)).dot(U)
        self.PtPU = PtPU
        (N,K) = Phi.shape            

        # Preprocess
        PtP = (Phi.T).dot(Phi) # FTF in Geoff's code
        assert((K,K) == PtP.shape)
        self.PtPUP = (PtPU).dot(Phi)
        assert((K,K) == self.PtPUP.shape)        
        self.PtPU_P = (PtPU - Phi.T)
        # FMI in Geoff's codes
        assert((K,N) == self.PtPU_P.shape)            
        self.PtP = PtP

        # Get dense versions. TODO: only if sparsity is low?
        self.Phi_dense = Phi.toarray()
        self.PtP_dense = PtP.toarray()
        self.PtPU_dense = PtPU.toarray()
        self.PtPUP_dense = self.PtPUP.toarray()
        self.PtPU_P_dense =self.PtPU_P.toarray()

    def form_Gh(self,x,y,**kwargs):
        """
        Form the G matrix and h vector from DENSE 
        matices.
        """
        sparse = kwargs.get('sparse',False)
        
        Ptq = self.Ptq
        if sparse:
            Phi = self.Phi
            PtPU_P = self.PtPU_P
            PtPU = self.PtPU
            PtPUP = self.PtPUP

            iXY = sps.diags(1.0 / (x+y),0)
            Y = sps.diags(y,0)
        else:
            Phi = self.Phi_dense
            PtPU_P = self.PtPU_P_dense
            PtPU = self.PtPU_dense
            PtPUP = self.PtPUP_dense
            Ptq = self.Ptq

            # Use broadcasting
            iXY = 1.0 / (x+y)
            Y = y
        
        # Eliminate both the x and y blocks
        (N,k) = Phi.shape
        
        g = self.sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        assert((N,) == g.shape)

        iXY = sps.diags(1.0 / (x+y),0)
        A = PtPU_P * iXY
        assert((k,N) == A.shape)
        
        G = ((A * Y).dot(Phi) - PtPUP)
        assert((k,k) == G.shape)

        h = A.dot(g) + PtPU.dot(x) + Ptq - Phi.T.dot(y)
        assert((k,) == h.shape)

        if sparse:
            G+= 1e-15*sps.eye(k)
        else:
            G += 1e-15*np.eye(k)
        
        return (G,g,h)

    def form_M(self):
        # Form M; expensive
        if hasattr(self,'M'):
            return self.M
        P = self.Phi
        U = self.U
        (N,K) = P.shape

        I = sps.eye(N)
        M = P.dot(U) + (I - P.dot(P.T)) # assumes P orthogonal
        self.M = M
        return M
        
    def form_full_newton(self,x,y):
        Phi = self.Phi
        (N,K) = Phi.shape
        
        X = sps.diags(x,0)
        Y = sps.diags(y,0)
        I = sps.eye(N)

        M = self.form_M()
        A = sps.bmat([[Y , X, None],
                      [-M, I, None],
                      [-I, I, Phi]],format='csr')
        assert((3*N,2*N+K) == A.shape)
        
        g = self.sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        r = M.dot(x) + self.q - y
        z = np.zeros(N)
        b = np.concatenate([g,r,z])
        assert((3*N,) == b.shape)

        return (A,b)
        
        
