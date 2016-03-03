import time
import os
import copy
import math

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg
import scipy

import matplotlib.pyplot as plt

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

    def get_iteration(self):
        return self.iterator.iteration
        
    def solve(self):
        """
        Call this to use an iteration object to solve an LCP
        
        The iteration solve should have all the LCP information
        Returns a record object.
        """
        assert(len(self.termination_conditions) >= 1)

        # Clean out any state from recorders
        for recorder in self.recorders:
            recorder.reset()
                
        while True:       
            # First record everything pertinent (record initial information first)
            for recorder in self.recorders:                    
                recorder.report(self.iterator)
                
            # Make any announcements
            for note in self.notifications:
                note.announce(self.iterator)
                
            # Then check for termination conditions
            for term_cond in self.termination_conditions:
                if term_cond.isdone(self.iterator):
                    print 'Termination reason:', term_cond
                    return                   
                
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

class IPIterator(Iterator):
    def get_step_len(self):
        raise NotImplementedError()
    def get_dir(self):
        raise NotImplementedError()
    def get_newton_system(self):
        raise NotImplementedError()
        
class MDPIterator(Iterator):
    def get_value_vector(self):
        raise NotImplementedError()

class PolicyIterator(Iterator):
    def get_policy_vector(self):
        raise NotImplementedError()

class BasisIterator(Iterator):
    def update_basis(self):
        raise NotImplementedError()
