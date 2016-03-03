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
    

def build_projlcp_from_basis_list(Qs,mdp_obj,flow_reg):
    # Find the in-basis approximations for E = I - gamma* P
    # Done with least-squares:
    # E   ~= Phi U
    # E.T ~= Phi W

    Phi = sps.block_diag(Qs)
    lcp_obj = mdp_obj.build_lcp(flow_regularization=flow_reg)
    M = lcp_obj.M
    q = lcp_obj.q
    (N,) = q.shape
    assert((N,N) == M.shape)
    assert(N == Phi.shape[0])
    
    PtPU = (Phi.T).dot(M)
    return (Phi,PtPU,q)
    

def OLD_build_projlcp_from_basis_list(Qs,mdp_obj,flow_reg):
    n = mdp_obj.num_states
    A = mdp_obj.num_actions
    N = (A+1)*n
    
    basis_sizes = [Qs[i].shape[1] for i in xrange(A+1)]
    K = sum(basis_sizes)

    BigU_blocks = [[None]]
    Uts = []
    (_,k) = Qs[0].shape
    for a in xrange(A):
        # Approximate E and E.T via least-squares
        E = mdp_obj.get_action_matrix(a)
        (_,d) = Qs[a+1].shape
        
        U = linalg.lsmr_matrix(Qs[0],E)
        W = linalg.lsmr_matrix(Qs[a+1],E.T)
        assert((k,n) == U.shape)
        assert((d,n) == W.shape)
        
        BigU_blocks[0].append(U) # [0, U1, U2,...]
        BigU_blocks.append([-W]\
                           + [None]*a\
                           + [flow_reg * Qs[a+1].T]\
                           + [None]*(A-a-1))
        # [-W, 0,...,flow_reg*Phi.T,0...]

    # Build the q vector
    q = np.hstack([-mdp_obj.state_weights]+mdp_obj.costs)
        
    # Construct the block basis:
    Phi = sps.block_diag(Qs)
    assert((N,K) == Phi.shape)

    # Construct the block coefficients:
    BigU = sps.bmat(BigU_blocks)
    print Phi.shape
    print BigU.shape
    assert((K,N) == BigU.shape)
    PtP = (Phi.T).dot(Phi)
    assert((K,K) == PtP.shape)
    PtPU = PtP.dot(BigU)

    return (Phi,PtPU,q)
                    
class ProjectiveGenerator(config.SolverGenerator):
    def __init__(self,**kwargs):
        # Parsing
        parser = KwargParser()
        parser.add('value_regularization',1e-12)
        parser.add('flow_regularization',1e-12)
        parser.add('termination_conditions')
        parser.add('recorders')
        parser.add_optional('notifications')
        parser.add('basis_generators')
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

        Qs = []
        for BG in self.basis_generators:
            Q = BG.generate_basis(points=points,
                                  mdp_obj=mdp_obj,
                                  discretizer=discretizer,
                                  special_points=special_points)
            Qs.append(Q)
        #visualize_bases(Qs,A,n,lens)

        f_reg = self.flow_regularization
        (Phi,PtPU,q) = build_projlcp_from_basis_list(Qs,
                                                     mdp_obj,
                                                     f_reg)

        proj_lcp_obj = lcp.ProjectiveLCPObj(Phi,PtPU,q)
        lcp_obj = mdp_obj.build_lcp()
        utils.banner('Remove access to MDP and LCP when using sketch')
        iter = ProjectiveIPIterator(proj_lcp_obj,
                                    mdp_obj=mdp_obj,
                                    lcp_obj=lcp_obj)
        objects = {'mdp':mdp_obj,
                   'proj_lcp':proj_lcp_obj}

        solver = solvers.IterativeSolver(iter)
        config.add_trn(self,solver) # Termination, Recording, and Notify      
    
        return [solver,objects]
    
    def extract(self,solver):               
        return config.basic_extract(self,solver)
