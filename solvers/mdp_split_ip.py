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
        self.discount = mdp_obj.discount
        self.state_weight_vector = np.ones(n) # `p'

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

        self.x = kwargs.get('x0',np.ones(N))
        self.y = kwargs.get('y0',np.ones(N))
        self.w = kwargs.get('w0',np.zeros(K))

        self.iteration = 0
        
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
        
        q = self.build_q()
        assert((N,) == q.shape)
        
        self.proj_lcp_obj = lcp.ProjectiveLCPObj(BigQ,U,q)
        return self.proj_lcp_obj

    def build_q(self):
        Q = self.Q
        N = self.num_nodes
        K = self.num_bases
        n = self.num_states
        A = self.num_actions
        
        Pi = self.get_projection_function()
        p = self.state_weight_vector
        q = np.empty(N)

        gamma = self.discount

        v_slice = self.get_slice(0)
        q[v_slice] = -Pi(p)
        v = self.x[v_slice]
        
        for a in xrange(A):
            f_slice = self.get_slice(a+1)
            f = self.x[f_slice]
            
            c = self.mdp_obj.costs[a]
            P = self.mdp_obj.transitions[a]

            q[v_slice] -= Pi(gamma * P.dot(f))

            q[f_slice] = Pi(c + gamma * (P.T).dot(v))
            
        return q
        
    def get_projection_function(self):
        Q = self.Q
        return lambda x: Q.dot((Q.T).dot(x))
        
    def get_slice(self,i):
        n = self.num_states
        A = self.num_actions
        assert(0 <= i < (A + 1))
        return slice(n*i,n*(i+1))        
        
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
        q_new = self.build_q()
        print 'q vector change:', np.linalg.norm(q_new - q)
        self.proj_lcp_obj.q = q_new

    def get_primal_vector(self):
        return self.x
    def get_dual_vector(self):
        return self.y
    def get_iteration(self):
        return self.iteration
