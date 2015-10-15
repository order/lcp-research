import numpy as np
import time
import os
import copy
import math

import scipy.sparse as sps
import scipy.sparse.linalg
import scipy
import matplotlib.pyplot as plt
import math
from util import *
import mdp
import lcp

"""
This file contains a number of different iterative solvers for linear complementary problems.
Most are phrased as iteration generators. These plug in to the iter_solver class, which wraps
the generators and provides a standardized framework for recording per-iteration information (like residuals and state) and termination checking (like maximum iteration count or residual threshold checking.
"""

class IterativeSolver(object):

    """
    This is the generic framework for iterative solvers.
    
    Basically does all the scaffolding stuff for the iteration including
    mantaining a list of "recorders" and checking a list of termination conditions.
    """

    def __init__(self,iterator):
        self.recorders = []
        self.termination_conditions = []
        self.iterator = iterator
        self.notifications = []
        
    def solve(self):
        """
        Call this to use an iteration object to solve an LCP
        
        The iteration solve should have all the LCP information
        Returns a record object.
        """
        assert(len(self.termination_conditions) >= 1)
        
        Records = [[] for _ in xrange(len(self.recorders))]
        
        while True:       
            # First record everything pertinent (record initial information first)
            for (i,recorder) in enumerate(self.recorders):                    
                Records[i].append(recorder.report(self.iterator))
                
            # Make any announcements
            for note in self.notifications:
                note.announce(self.iterator)
                
            # Then check for termination conditions
            for term_cond in self.termination_conditions:
                if term_cond.isdone(self.iterator):
                    print 'Termination reason:', term_cond
                    return Records                   
                
            # Finally, advance to the next iteration
            self.iterator.next_iteration()
                
class Iterator(object):
    """
    Abstract definition of an iterator
    """
    def next_iteration(self):
        raise NotImplementedError()
    def get_iteration(self):
        raise NotImplementedError()
        
class LCPIterator(Iterator):
    def get_primal_vector(self):
        raise NotImplementedError()
    def get_dual_vector(self):
        raise NotImplementedError()
    def get_gradient_vector(self):
        raise NotImplementedError()

        
class MDPIterator(Iterator):
    def get_value_vector(self):
        raise NotImplementedError()
    
class ValueIterator(MDPIterator):
    def __init__(self,mdp_obj,**kwargs):
        self.mdp = mdp_obj
        self.iteration = 0
        
        N = mdp_obj.num_states
        A = mdp_obj.num_actions
        
        self.v = kwargs.get('v0',np.ones(N))
        self.costs = mdp_obj.costs
        self.PT = [mdp_obj.discount * x.transpose(True) for x in mdp_obj.transitions]
        
        #self.pool = multiprocessing.Pool(processes=4)
        
    def next_iteration(self): 
        """
        Do basic value iteration, but also try to recover the flow variables.
        This is for comparison to MDP-split lcp_obj solving (see mdp_ip_iter)
        """        
        A = self.mdp.num_actions
        N = self.mdp.num_states
        
        #M = [self.pool.apply(update, args=(c,PT,self.v))\
        #    for (c,PT) in zip(self.costs,self.PT)]
        
        Vs = np.empty((N,A))
        for a in xrange(A):     
            Vs[:,a] = self.mdp.costs[a] + self.PT[a].dot(self.v)
            #assert(not np.any(np.isnan(Vs[:,a])))
        assert((N,A) == Vs.shape)
    
        self.v = np.amin(Vs,axis=1)
        assert((N,) == self.v.shape)
        self.iteration += 1
       
    def get_value_vector(self):
        return self.v
        
    def get_iteration(self):
        return self.iteration
