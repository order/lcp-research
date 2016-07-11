from solvers import *

import numpy as np
import scipy as sp
import scipy.sparse as sps

import utils
import matplotlib.pyplot as plt

from collections import defaultdict

import time
    
###################################################################
# Start of the iterator
class ProjectiveIPIterator(LCPIterator,IPIterator,BasisIterator):
    def __init__(self,plcp,**kwargs):
        self.mdp_obj = kwargs.get('mdp_obj',None)
        self.lcp_obj = kwargs.get('lcp_obj',None)
                
        self.plcp = plcp
        
        Phi = plcp.Phi
        U = plcp.U
        PtPU = plcp.PtPU
        q = plcp.q
        (N,K) = Phi.shape
        assert((K,N) == PtPU.shape)

        self.q = plcp.q
        self.Ptq = Phi.T.dot(self.q)
        assert((K,) == self.Ptq.shape)
        
        self.update_P_PtPU(Phi,U,PtPU)

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',Phi.T.dot(q))
        w_res = self.x - self.y + q - Phi.dot(self.w)
        print 'W residual', np.linalg.norm(w_res)
        assert(np.linalg.norm(w_res) < 1e-15)

        self.sigma = 0.95
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
        #self.data['x'].append(x)
        #self.data['y'].append(y)
        #self.data['w'].append(w)
        

        (P,ip_term,x_term,y_term) = potential(x,y,np.sqrt(N))
        self.data['P'].append(P)

        dot = x.dot(y)

        self.data['ip'].append(dot / float(N))

        # Step 3: form right-hand sides
        sigma = self.sigma
                
        # Step 4: Form the reduced system G dw = h
        (G,g,h) =  self.form_Gh(x,y)     

        if isinstance(G,sps.spmatrix):
            fillin = float(G.nnz) / float(G.size)
        else:
            fillin = 1.0

        # Step 5: Solve for dir_w
        if fillin > 0.1:
            if isinstance(G,sps.spmatrix):
                G = G.toarray()
            dir_w = np.linalg.solve(G + 1e-12*np.eye(k),h)
        else:
            dir_w = sps.linalg.spsolve(G + 1e-12*sps.eye(k),h)
        assert((k,) == dir_w.shape)
            
            # Step 6
        Phidw= Phi.dot(dir_w)
        assert((N,) == Phidw.shape)
        S = x+y
       
        inv_XY = sps.diags(1.0/S,0)
        dir_y = inv_XY.dot(g - y*Phidw)
        assert((N,) == dir_y.shape)
        
        # Step 7
        dir_x = dir_y + Phidw
        assert((N,) == dir_x.shape)

        print '||dx||:',np.linalg.norm(dir_x)
        print '||dy||:',np.linalg.norm(dir_y)
        print '||dw||:',np.linalg.norm(dir_w)
        self.data['dx'].append(np.linalg.norm(dir_x))
        self.data['dy'].append(np.linalg.norm(dir_y))
        self.data['dw'].append(np.linalg.norm(dir_w))
        
        # Get step length so that x,y are strictly feasible after move
        steplen = steplen_heuristic(x,dir_x,y,dir_y,0.6)
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
        self.sigma = sigma          
              
        self.x = x + steplen * dir_x
        self.y = y + steplen * dir_y
        self.w = w + steplen * dir_w
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_w = dir_w

        w_res = dir_x - dir_y - Phidw
        print 'W residual:', np.linalg.norm(w_res)
        
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
        self.PtPU_dense = PtPU.toarray()
        self.PtPUP_dense = self.PtPUP.toarray()
        self.PtPU_P_dense =self.PtPU_P.toarray()

    def form_Gh(self,x,y,sparse=False):
        """
        Form the G matrix and h vector from DENSE 
        matices.
        """
        
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
        
        return (G,g,h)
