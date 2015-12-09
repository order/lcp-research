from termination import MaxIterTerminationCondition,ResidualTerminationCondition
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
        n = mdp_obj.num_states
        A = mdp_obj.num_actions

        # Store basis
        assert(n == Phi.shape[0])
        assert(2 == len(Phi.shape))

        k = Phi.shape[1]
        self.Phi = Phi

        # Store sizes
        N = (A+1)*n
        K = (A+1)*k

        self.num_states = n
        self.num_actions = A
        self.num_nodes = N
        self.num_bases = K

        
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
        N = self.num_nodes
        K = self.num_bases
        A = self.num_actions
        
        # Block diagonal basis
        RepQ = [Q]*(A+1)
        if isinstance(Q, np.ndarray):
            BigQ = sp.linalg.block_diag(*RepQ)
        else:
            BigQ = sps.block_diag(RepQ)
        assert((N,K) == BigQ.shape)
        self.BigQ = BigQ
        
        # Build the U part of the matrix
        U = mdp.mdp_skew_assembler([Q.T]*A)
        
        assert((N,N) == U.shape)
        self.U = U
        
        proj_q = self.build_original_q()
        assert((N,) == proj_q.shape)        
        
        self.proj_lcp_obj = lcp.ProjectiveLCPObj(BigQ,U,proj_q)
        return self.proj_lcp_obj
        
    
    def build_original_q(self):
        """
        Build the original projected q
        Under the assumption that v,f = 0
        """
        Q = self.Q
        N = self.num_nodes
        K = self.num_bases
        n = self.num_states
        A = self.num_actions
        
        Pi = self.get_projection_function()
        q_comps = [-Pi(np.ones(n))] # State-action weights
        for a in xrange(A):
            # Cost components
            q_comps.append(Pi(self.mdp_obj.costs[a]))
            
        proj_q = np.hstack(q_comps)
        assert((N,) == proj_q.shape)
        self.original_proj_q = proj_q
        return proj_q
        
    def get_projection_function(self):
        Q = self.Q
        return lambda x: Q.dot((Q.T).dot(x))
        
    def get_slice(self,i):
        n = self.num_states
        A = self.num_actions
        assert(0 <= i < (A + 1))
        return slice(n*i,n*(i+1))
        
    def update_q(self):
        n = self.num_states
        A = self.num_actions
        Q = self.Q
        
        Pi = self.get_projection_function()
        
        # Build difference vector
        delta = []
        v_block = np.zeros(n)
        v_slice = self.get_slice(0)
        v = self.x[v_slice]
        assert((n,) == v.shape)
        for a in xrange(A):
            P = self.mdp_obj.transitions[a]
            
            # Aggregate the changes to v block
            f_slice = self.get_slice(a+1)
            f = self.x[f_slice]
            v_block += Pi(P.dot(f))

            # Append the f_a block to the list
            delta.append(Pi((P.T).dot(v)))
            
        # Prepend the v_block
        delta.insert(0,v_block)
        assert(A+1 == len(delta))
        
        # Convert to vector
        delta = np.hstack(delta)
        assert((n*(A+1),) == delta.shape)
        
        # Update the projective LCP
        gamma = self.mdp_obj.discount
        new_q = self.original_proj_q + gamma*delta
        self.proj_lcp_obj.update_q(new_q)
        
        # Return the new vector
        return self.proj_lcp_obj
        
        
    def next_iteration(self):
        N = self.num_nodes
        K = self.num_bases
        n = self.num_states
        A = self.num_actions
        
        assert((N,) == self.x.shape)
        assert((N,) == self.y.shape)
        assert((K,) == self.w.shape)

        print 'Creating a new projective IP iterator'
        inner_solver = ProjectiveIPIterator(self.proj_lcp_obj,
            x0=self.x,
            y0=self.y,
            w0=self.w)
        term_conds = [MaxIterTerminationCondition(500),
                      ResidualTerminationCondition(1e-4)]

        done_flag = False
        while not done_flag:    
            inner_solver.next_iteration()            

            for cond in term_conds:
                if cond.isdone(inner_solver):
                    print 'Inner term condition:', cond
                    self.x = inner_solver.x
                    self.y = inner_solver.y
                    self.w = inner_solver.w
                    assert((N,) == self.x.shape)
                    assert((N,) == self.y.shape)
                    assert((K,) == self.w.shape)

                    done_flag = True
                    break
                
        self.iteration += 1
        q = self.proj_lcp_obj.q
        self.update_q()
        print 'q vector change:', np.linalg.norm(q -self.proj_lcp_obj.q)

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
