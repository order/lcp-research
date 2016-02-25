
import solvers
from solvers.projective import ProjectiveIPIterator
from solvers.termination import *
from solvers.notification import *
from solvers.recording import *
import config
import bases
from bases.svd import SVDBasis
import linalg
import lcp
import utils
from utils.parsers import KwargParser

import numpy as np
import scipy as sp
import scipy.sparse as sps
import matplotlib.pyplot as plt

import time

def visualize_bases(Qs,A,n,lens):
    assert(A+1 == len(Qs))
    (_,K) = Qs[0].shape
    Q_stack = np.vstack(Qs)    
    
    utils.banner('Visualizing bases')
    for k in xrange(K):
        v = (Q_stack[:,k])[np.newaxis,:]
        Frames = utils.processing.split_into_frames(v,A,n,lens)
        f,axarr = plt.subplots(1,A+1)
        for i in xrange(A+1):
            axarr[i].pcolormesh(Frames[0,i,:,:])
        plt.show()

def build_projlcp_from_value_basis(Q,mdp_obj,flow_reg):
    A = mdp_obj.num_actions
    return build_projlcp_from_basis_list([Q]*(A+1),
                                         mdp_obj,
                                         flow_reg)

def build_projlcp_from_basis_list(Qs,mdp_obj,flow_reg):
    # Find the in-basis approximations for E = I - gamma* P
    # Done with least-squares:
    # E   ~= Phi U
    # E.T ~= Phi W
    n = mdp_obj.num_states
    A = mdp_obj.num_actions
    N = (A+1)*n
    k = Qs[0].shape[1]
    K = (A+1)*k
    assert(k <= n)

    for Q in Qs:
        assert((n,k) == Q.shape)
    
    BigU_blocks = [[None]]
    Uts = []
    for a in xrange(A):
        # Approximate E and E.T via least-squares
        E = mdp_obj.get_action_matrix(a)
        U = linalg.lsmr_matrix(Qs[a],E)
        assert((k,n) == U.shape)

        W = linalg.lsmr_matrix(Qs[a],E.T)
        assert((k,n) == W.shape)

        BigU_blocks[0].append(U) # [0, U1, U2,...]
        BigU_blocks.append([-W]\
                           + [None]*a\
                           + [flow_reg * Qs[a].T]\
                           + [None]*(A-a-1))
        # [-W, 0,...,flow_reg*Phi.T,0...]

    # Build the q vector
    q = np.hstack([-mdp_obj.state_weights]+mdp_obj.costs)
        
    # Construct the block basis:
    BigQ = sps.block_diag(Qs)
    assert((N,K) == BigQ.shape)

    # Construct the block coefficients:
    BigU = sps.bmat(BigU_blocks)
    assert((K,N) == BigU.shape)

    return (BigQ,BigU,q)
                    
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
        lens = discretizer.get_basic_lengths()
        xy = np.prod(lens) # Physical size

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

        svd = True
        if svd:
            print N
            print np.sqrt(N)
            num_svd = 10
            utils.banner('Using SVD basis; split out a make more principled')
            BG = SVDBasis(num_svd)
            Qs = BG.generate_basis(mdp_obj=mdp_obj,
                                   discretizer=discretizer)
            visualize_bases(Qs,A,n,lens)
                                    
        else:
            raise NotImplementedError()
            BG = self.basis_generator
            Q = BG.generate_basis(points,
                                  special_points=special_points)

        (BigQ,BigU,q) = build_projlcp_from_basis_list(Qs,
                                                      mdp_obj,
                                                      self.flow_regularization)

        proj_lcp_obj = lcp.ProjectiveLCPObj(BigQ,BigU,q)
        iter = ProjectiveIPIterator(proj_lcp_obj)
        objects = {'mdp':mdp_obj,
                   'proj_lcp':proj_lcp_obj}

        solver = solvers.IterativeSolver(iter)
        config.add_trn(self,solver) # Termination, Recording, and Notify      
    
        return [solver,objects]
    
    def extract(self,solver):               
        return config.basic_extract(self,solver)
