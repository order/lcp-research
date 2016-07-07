from solvers import *

import numpy as np
import scipy as sp
import scipy.sparse as sps

import utils
import matplotlib.pyplot as plt

from collections import defaultdict

import time

def form_Gh_dense(x,y,w,g,p,q,Phi,PtPUP,PtPU_P):
    """
    Form the G matrix and h vector from DENSE 
    matices.
    """
    start = time.time()
    for v in [Phi,PtPUP,PtPU_P]:
        assert(type(v) == np.ndarray)
        
    # Eliminate both the x and y blocks
    (k,N) = PtPU_P.shape

    A = (PtPU_P / (x+y))
    assert((k,N) == A.shape)
    G = ((A * y).dot(Phi) - PtPUP)
    assert((k,k) == G.shape)
    
    h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
    assert((k,) == h.shape)
    #print '\t(G,h) dense construction time:',time.time() - start
    return (G,h)

def form_Gh_sparse(x,y,w,g,p,q,Phi,PtPUP,PtPU_P):
    """
    Form the G matrix and h vector from SPARSE 
    matices.
    """
    start = time.time()

    for A in [Phi,PtPUP,PtPU_P]:
        assert(isinstance(A,sps.spmatrix))
    (k,N) = PtPU_P.shape

    # Eliminate both the x and y blocks
    Y = sps.spdiags(y,0,N,N)
    iXY = sps.spdiags(1.0 / (x+y),0,N,N)
    A = PtPU_P.dot(iXY)
    assert((k,N) == A.shape)
    G = ((A.dot(Y)).dot(Phi) - PtPUP)
    assert((k,k) == G.shape)
    
    h = A.dot(g + y*p) - PtPU_P.dot(q - y) + PtPUP.dot(w)
    assert((k,) == h.shape)
    #print '\t(G,h) sparse construction time:',time.time() - start

    return (G,h)
    
###################################################################
# Start of the iterator
class ProjectiveIPIterator(LCPIterator,IPIterator,BasisIterator):
    def __init__(self,proj_lcp_obj,**kwargs):
        self.centering_coeff = kwargs.get('centering_coeff',0.99)
        self.linesearch_backoff = kwargs.get('linesearch_backoff',
                                             0.99)
        self.central_path_backoff = kwargs.get('central_path_backoff',0.9995)
        self.mdp_obj = kwargs.get('mdp_obj',None)
        self.lcp_obj = kwargs.get('lcp_obj',None)
        
        self.proj_lcp_obj = proj_lcp_obj
        self.q = proj_lcp_obj.q
        Phi = proj_lcp_obj.Phi
        U = proj_lcp_obj.U
        PtPU = proj_lcp_obj.PtPU
        (N,K) = Phi.shape
        assert((K,N) == PtPU.shape)
        
        self.update_P_PtPU(Phi,U,PtPU)

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(K))
        self.steplen = np.nan
        self.dir_x = np.full(N,np.nan)
        self.dir_y = np.full(N,np.nan)
        self.dir_w = np.full(K,np.nan)
        
        assert((N,) == self.x.shape)
        assert((N,) == self.y.shape)
        assert((K,) == self.w.shape)

        self.data = defaultdict(list)

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
        #assert((k,N) == U.shape)
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
        #self.data['x'].append(x)
        #self.data['y'].append(y)
        #self.data['w'].append(w)
        

        (P,ip_term,x_term,y_term) = potential(x,y,np.sqrt(N))
        self.data['P'].append(P)

        # Use M = PU + (I - PP^T)
        recovered_y = Phi.dot(U.dot(x))\
                      + x - Phi.dot(Phi.T.dot(x)) + q
        
        r = recovered_y - y 
        dot = x.dot(y)

        self.data['res_norm'].append(np.linalg.norm(r))
        self.data['ip'].append(dot / float(N))

        # Step 3: form right-hand sides
        g = sigma * x.dot(y) / float(N) * np.ones(N) - x * y
        p = x - y + q - Phi.dot(w)
        assert((N,) == g.shape)
        assert((N,) == p.shape)
        
        # Step 4: Form the reduced system G dw = h
        if False:
            (G,h) =  form_Gh_dense(x,y,w,g,p,q,self.Phi_dense,
                                   self.PtPUP_dense,
                                   self.PtPU_P_dense)
        else:
            (G,h) =  form_Gh_sparse(x,y,w,g,p,q,self.Phi,
                                   self.PtPUP,
                                   self.PtPU_P)        

        if isinstance(G,sps.spmatrix):
            fillin = float(G.nnz) / float(G.size)
        else:
            fillin = 1.0
        #print 'Fill in:',fillin

        # Step 5: Solve for del_w
        start = time.time()
        if fillin > 0.1:
            if isinstance(G,sps.spmatrix):
                G = G.toarray()
            del_w = np.linalg.solve(G + 1e-8*np.eye(G.shape[0]),h)
        else:
            del_w = sps.linalg.spsolve(G + 1e-8*sps.eye(G.shape[0]),h)
        assert((k,) == del_w.shape)
        #print 'Solve time:',time.time() - start
            
            # Step 6
        Phidw= Phi.dot(del_w)
        assert((N,) == Phidw.shape)
        S = x+y
        print 'min(S):',np.min(S)
        print 'max(S):',np.max(S)
        print 'argmin(S)',np.argmin(S)
        print 'argmax(S)',np.argmax(S)
        
        #self.data['sum'].append(S)
        #self.data['product'].append(x*y)

        
       
        inv_XY = sps.diags(1.0/S,0)
        del_y = inv_XY.dot(g + y*p - y*Phidw)
        assert((N,) == del_y.shape)
        
        # Step 7
        del_x = del_y+Phidw-p
        assert((N,) == del_x.shape)

        print '||dx||:',np.linalg.norm(del_x)
        print '||dy||:',np.linalg.norm(del_y)
        print '||dw||:',np.linalg.norm(del_w)
        self.data['dx'].append(np.linalg.norm(del_x))
        self.data['dy'].append(np.linalg.norm(del_y))
        self.data['dw'].append(np.linalg.norm(del_w))
        
           
        # Step 8 Step length for x and y separately
        x_mask = (del_x < 0) # Decreasing direction
        if np.any(x_mask):
            # Stop just short of  max step
            x_steplen = min(1.0, 0.1*np.min(x[x_mask] / -del_x[x_mask]))
            print 'Argmin steplen', np.argmin(x[x_mask] / -del_x[x_mask])
        else:
            x_steplen = 1.0
          
        y_mask = (del_y < 0) # Decreasing direction
        if np.any(y_mask):
            # Stop just short of  max step
            y_steplen = min(1.0, 0.25*np.min(y[y_mask] / -del_y[y_mask])) 
        else:
            y_steplen = 1.0

        least_steplen = min(x_steplen,y_steplen)
        print 'Steplen',least_steplen

        # Sigma is beta in Geoff's code
        if(1.0 >= least_steplen > 0.95):
            sigma *= 0.99 # Long step
        elif(0.05 >= least_steplen > 1e-3):
            sigma *= 1.05
            sigma = min(sigma,0.99)
        elif (least_steplen <= 1e-3):
            sigma = 0.99 # Short step
        print 'Sigma',sigma
        #self.data['sigma'].append(sigma)

        # Update point and fields
        self.centering_coeff = sigma
              
        self.x = x + least_steplen * del_x
        self.y = y + least_steplen * del_y
        self.w = w + least_steplen * del_w
        self.dir_x = del_x
        self.dir_y = del_y
        self.dir_w = del_w
        self.iteration += 1

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

    def update_basis(self,basis_fn,block_id):
        Phi = self.Phi
        (N,K) = Phi.shape
        (n,) = basis_fn.shape
        assert(0 == N % n)
        Aplus1 = N / n

        # Assuming sparse Phi
        assert(isinstance(Phi,sps.spmatrix))
        row = np.arange(block_id*n,(block_id+1)*n)
        col = np.zeros(n)
        basis_vector = sps.coo_matrix((basis_fn,(row,col)),
                                      shape=(N,1))

        # Better way of doing this to maintain block sparsity
        Phi = sps.hstack([basis_vector,Phi])
        M = self.lcp_obj.M
        
        PtPU = (Phi.T).dot(M)
        self.update_P_PtPU(Phi,PtPU)
        

    def update_P_PtPU(self,Phi,U,PtPU):        
        self.Phi = Phi
        self.U = U
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
        self.PtPUP_dense = self.PtPUP.toarray()
        self.PtPU_P_dense =self.PtPU_P.toarray()
