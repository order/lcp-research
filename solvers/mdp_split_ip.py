from termination import MaxIterTerminationCondition
from solvers import MDPIterator
from projective import ProjectiveIPIterator
import mdp
import lcp

import numpy as np
import scipy as sp
import scipy.sparse as sps

import matplotlib.pyplot as plt

class MDPSplitIPIterator(MDPIterator):
    def __init__(self,mdp_obj,Phi,**kwargs):
    
        ortho = kwargs.get('orthogonal',False)
    
        # Store mdp
        self.mdp_obj = mdp_obj
        N = mdp_obj.num_states
        A = mdp_obj.num_actions
        self.num_states = N
        self.num_actions = A
        
        # Store basis
        assert(N == Phi.shape[0])
        assert(2 == len(Phi.shape))

        K = Phi.shape[1]
        self.num_bases = K
        self.Phi = Phi
        
        # Orthogonalize the basis if not already orthogonal
        if ortho:
            self.Q = Phi
        else:
            assert(isinstance(Phi,np.ndarray)) # Cannot be sparse
            """
            There isn't a sparse QR decomposer.
            Fill-in would probably make this pointless anyhow.
            """
            
            [Q,R] = sp.linalg.qr(Phi,mode='economic')
            assert(Q.shape == Phi.shape)
            self.Q = Q
        
        # Build all the components of the projective LCP
        self.build_projective_lcp()
        
        # Build the projective iterator
        self.proj_iter = ProjectiveIPIterator(self.proj_lcp_obj)

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(K))

        self.iteration = 0

        
    def build_projective_lcp(self):
        """
        Build the projective lcp
        """
        
        Q = self.Q
        A = self.num_actions
        N = self.num_states
        
        # Block diagonal basis
        RepQ = [Q]*(A+1)
        if isinstance(Q, np.ndarray):
            BigQ = sp.linalg.block_diag(*RepQ)
        else:
            BigQ = sps.block_diag(RepQ)
        assert((A+1)*N == BigQ.shape[0])
        self.BigQ = BigQ
        
        # Build the U part of the matrix
        BigN = N * (A+1)
        U = mdp.mdp_skew_assembler([Q.T]*A)
        
        assert((BigN,BigN) == U.shape)
        self.U = U
        
        proj_q = self.build_original_q()
        assert((BigN,) == proj_q.shape)        
        
        self.proj_lcp_obj = lcp.ProjectiveLCPObj(BigQ,U,proj_q)
        return self.proj_lcp_obj
        
    
    def build_original_q(self):
        """
        Build the original projected q
        Under the assumption that v,f = 0
        """
        Q = self.Q
        N = self.num_states
        A = self.num_actions
        
        Pi = self.get_projection_function()
        q_comps = [-Pi(np.ones(N))]
        for a in xrange(A):
            q_comps.append(Pi(self.mdp_obj.costs[a]))
            
        proj_q = np.hstack(q_comps)
        assert((N*(A+1),) == proj_q.shape)
        self.original_proj_q = proj_q
        return proj_q
        
    def get_projection_function(self):
        Q = self.Q
        return lambda x: Q.dot((Q.T).dot(x))
        
    def get_slice(self,i):
        N = self.num_states
        A = self.num_actions
        assert(0 <= i < (A + 1))
        return slice(N*i,N*(i+1))
        
    def update_q(self):
        N = self.num_states
        A = self.num_actions
        Q = self.Q
        
        Pi = self.get_projection_function()
        
        # Build difference vector
        delta = []
        v_block = np.zeros(N)
        v_slice = self.get_slice(0)
        v = self.x[v_slice]
        assert((N,) == v.shape)
        for a in xrange(A):
            P = self.mdp_obj.transitions[a]
            
            # Aggregate the changes to v block
            f_slice = self.get_slice(a+1)
            f = self.x[f_slice]
            v_block += Pi(P.dot(f))

            # Append the f_a block to the list
            delta.append(PI((P.T).dot(v)))
            
        # Prepend the v_block
        delta.insert(0,v_block)
        assert(A+1 == len(delta))
        
        # Convert to vector
        delta = np.hstack(delta)
        assert((N*(A+1),) == delta.shape)
        
        # Update the projective LCP
        gamma = self.mdp_obj.discount
        new_q = self.original_proj_q + gamma*delta
        self.proj_lcp_obj.update_q(new_q)
        
        # Return the new vector
        return self.proj_lcp_obj
        
        
    def next_iteration(self):
        N,A,K = self.num_states,self.num_actions,self.num_bases
        
        # Don't create every time?
        inner_solver = ProjectiveIPIterator(self.proj_lcp_obj,
            x0=self.x,
            y0=self.y,
            w0=self.w)
        inner_term_cond = MaxIterTerminationCondition(500)
        
        while True:    
            print 'Inner iteration:',inner_solver.iteration
            # Then check for termination conditions
            if inner_term_cond.isdone(inner_solver):
                print 'Inner term condition:', term_cond
                self.x = inner_solver.x
                self.y = inner_solver.y
                self.w = inner_solver.w
                return 
            # Finally, advance to the next iteration
            inner_solver.next_iteration()            
        self.iteration += 1

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
