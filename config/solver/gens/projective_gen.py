from utils.parsers import KwargParser

import solvers
from solvers.projective import ProjectiveIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *
import config
import bases
import linalg
import lcp

import numpy as np
import scipy as sp
import scipy.sparse as sps

import time

class ProjectiveGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('value_regularization',1e-12)
        parser.add('flow_regularization',1e-12)
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        parser.add('basic_basis_generator')
        args = parser.parse(kwargs)

        # Dump into self namespace
        self.__dict__.update(args)

    def generate(self,discretizer):
        mdp_obj = discretizer.build_mdp()

        # Get sizes
        A = mdp_obj.num_actions
        n = mdp_obj.num_states
        N = (A+1)*n
        gamma = mdp_obj.discount

        # Actual basis generator wraps the
        # basic basis generator
        # (Which doesn't deal with non-phyiscal states)
        basis_generator = bases.BasisGenerator(self.basic_basis_generator)

        points = discretizer.get_node_states()
        special_points = discretizer.get_special_node_indices()

        Phi = basis_generator.generate_basis(points,
                                             special_points=special_points)
        assert((n,n) == Phi.shape)
        Phi_other = np.eye(n)
        assert(np.linalg.norm(Phi - Phi_other) / n < 1e-12)
        (d,k) = Phi.shape
        assert(d == n)
        assert(k <= n)
        K = (A+1)*k

        #if basis_gen.isortho():
        Q = Phi
        #else:
            #Orthogonalize using QR decomposition
        #    [Q,R] = sp.linalg.qr(Phi,mode='economic')
        #    assert(Q.shape == Phi.shape)


        # Find the in-basis approximations for E = I - gamma* P
        # Done with least-squares:
        # E   ~= Phi U
        # E.T ~= Phi W
        
        BigU_blocks = [[None]]
        flow_reg = self.flow_regularization
        for a in xrange(A):
            P = mdp_obj.transitions[a]
            E = sps.eye(n) - gamma * P
            U = linalg.lsmr_matrix(Q,E)
            W = linalg.lsmr_matrix(Q,E.T)
            assert((k,n) == U.shape)
            assert((k,n) == W.shape)

            BigU_blocks[0].append(U) # [0, U1, U2,...]
            BigU_blocks.append([-W] + [None]*a + [flow_reg * Phi.T] +  [None]*(A-a-1))
            # [-W, 0,...,flow_reg*Phi.T,0...]

        # Build the q vector
        q = np.hstack([-mdp_obj.state_weights]+ mdp_obj.costs)
        

        # Construct the block basis:
        # Phi 0   ... 0
        # 0   Phi ... 0
        #     ...
        # 0   ...     Phi
        BigQ = sps.block_diag([Q]*(A+1))
        assert((N,K) == BigQ.shape)

        # Construct the block coefficients:
        # 0 U_1 ... U_A
        # -W_1  0 ... 0
        #         ...
        # -W_A  0 ... 0
        BigU = sps.bmat(BigU_blocks)
        assert((K,N) == BigU.shape)

        proj_lcp_obj = lcp.ProjectiveLCPObj(BigQ,BigU,q)
        iter = ProjectiveIPIterator(proj_lcp_obj)
        objects = {'mdp':mdp_obj,
                   'proj_lcp':proj_lcp_obj}

        solver = solvers.IterativeSolver(iter)
        config.add_trn(self,solver) # Termination, Recording, and Notify      
    
        return [solver,objects]
    
    def extract(self,solver):               
        return config.basic_extract(self,solver)
