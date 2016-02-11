
import solvers
from solvers.projective import ProjectiveIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *
import config
import bases
import linalg
import lcp
import utils
from utils.parsers import KwargParser

import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt

import time

def visualize_bases(Q,U,x,y,K):
    xy = x*y
    utils.banner('Visualizing bases')
    for i in xrange(K):
        f,(ax1,ax2) = plt.subplots(1,2)
        ax1.pcolor(np.reshape(Q[:xy,i],(x,y)))
        ax2.pcolor(np.reshape(U.T[:xy,i],(x,y)))
        plt.show()
                    
class ProjectiveGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('value_regularization',1e-12)
        parser.add('flow_regularization',1e-12)
        parser.add('x_dual_bases',False)
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        parser.add('basis_generator')
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
        points = discretizer.get_node_states()
        special_points = discretizer.get_special_node_indices()

        BG = self.basis_generator
        Phi = BG.generate_basis(points,
                                special_points=special_points)

        (d,k) = Phi.shape
        assert(d == n)
        assert(k <= n)
        K = (A+1)*k

        #Orthogonalize using QR decomposition
        #[Q,R] = sp.linalg.qr(Phi,mode='economic')
        #assert(Q.shape == Phi.shape)
        Q = Phi

        # Find the in-basis approximations for E = I - gamma* P
        # Done with least-squares:
        # E   ~= Phi U
        # E.T ~= Phi W
        
        BigU_blocks = [[None]]
        flow_reg = self.flow_regularization
        Uts = []
        for a in xrange(A):
            P = mdp_obj.transitions[a]
            E = sps.eye(n) - gamma * P
            U = linalg.lsmr_matrix(Q,E)
            assert((k,n) == U.shape)

            if not self.x_dual_bases:
                W = linalg.lsmr_matrix(Q,E.T)
                assert((k,n) == W.shape)

            #visualize_bases(Q,U,xn,yn,k)

            BigU_blocks[0].append(U) # [0, U1, U2,...]
            if self.x_dual_bases:
                Uts.append(U.T)
                BigU_blocks.append([-Phi.T]\
                                   + [None]*a\
                                   + [flow_reg * U]\
                                   + [None]*(A-a-1))
            else:
                BigU_blocks.append([-W]\
                                   + [None]*a\
                                   + [flow_reg * Phi.T]\
                                   + [None]*(A-a-1))
                # [-W, 0,...,flow_reg*Phi.T,0...]

        # Build the q vector
        q = np.hstack([-mdp_obj.state_weights]+ mdp_obj.costs)
        

        # Construct the block basis:
        # Phi 0   ... 0
        # 0   Phi ... 0
        #     ...
        # 0   ...     Phi
        if self.x_dual_bases:
            BigQ = sps.block_diag([Q] + Uts)
        else:
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
