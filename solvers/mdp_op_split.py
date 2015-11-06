import solver

from solvers import LCPIterator
from solvers import ProjectiveIPIterator
import mdp

import numpy as np
import scipy as sp
import scipy.sparse as sps

class MDPSplitIPIterator(LCPIterator):
    def __init__(self,mdp_obj,Phi,**kwargs):
    
        # Store mdp
        self.mdp_obj = mdp_obj
        N = mdp_obj.num_states
        A = mdp_obj.numactions
        self.N = N
        self.A = A
        
        # Store basis
        assert(N == Phi.shape[0])
        K = Phi.shape[1]
        self.K = K
        [Q,R] = sp.linalg.qr(Phi,mode='economic')
        assert(Q.shape == Phi.shape) # Was actually a basis
        self.Phi = Phi
        self.Q = Q     
            
        # Build the projective lcp        
        RepPhi = [Phi]*(A+1)
        BigPhi = scipy.linalg.block_diag(*RepPhi)
        U = mdp.build_proj_value_iter_matrix(N,A)
        proj_q = np.hstack([-np.ones(N)] + self.proj_costs)
        self.orig_proj_q = proj_q
        assert((N,) == q.shape)
        self.proj_lcp_obj = ProjectiveLCPObj(BigPhi,U,proj_q)
        
        # Build the projective iterator
        self.proj_iter = ProjectiveIPIterator(self.proj_lcp_obj)
        self.term_cond = PrimalChangeTerminationCondition(1e-5)

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(k))

        self.iteration = 0
        
    def update_q(self):
        proj_q = [-np.ones(N)]
        
    def next_iteration(self):
        N,A,K = self.N,self.A,self.K
        
        while not self.term_cond.isdone(self.proj_iter):
            self.proj_iter.next_iteration()
        self.x = self.proj_iter.get_primal_vector()
        self.y = self.proj_iter.get_dual_vector()
        
        gamma = self.mdp_obj.discount
        delta_r = []
        delta_p = np.zeros(N)
        for a in xrange(A):
            delta_p -= gamma * self.proj_trans.dot(
            
        
        self.iteration += 1

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
