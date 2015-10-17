import numpy as np

##########################
# Termination functions
class TerminationCondition(object):
    def isdone(self,iterator):
        raise NotImplementedError()
        
class ValueChangeTerminationCondition(TerminationCondition):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self,thresh):
        self.thresh = thresh
        self.old_v = np.NaN # Don't have old iteration
        self.diff = float('inf')
        
    def isdone(self,iterator):
        v = iterator.get_value_vector()
        if np.any(self.old_v == np.NaN):
            self.old_v = v
            return False
            
        self.diff = np.linalg.norm(self.old_v - v)
        self.old_v = v
        return self.diff < self.thresh
        
    def __str__(self):
        return 'ValueChangeTerminationCondition {0} ({1})'.format(self.thresh,self.diff)
        
class ResidualTerminationCondition(TerminationCondition):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self,thresh):
        self.thresh = thresh
        self.residual = None
        
    def isdone(self,iterator):
        x = iterator.get_primal_vector()
        w = iterator.get_dual_vector()
            
        self.residual = np.linalg.norm(np.minimum(x,w))
        return self.residual < self.thresh
        
    def __str__(self):
        return 'ResidualTerminationCondition {0} ({1})'.format(self.thresh,self.residual)   

class MaxIterTerminationCondition(TerminationCondition):
    def __init__(self,max_iter):
        self.max_iter = max_iter
    def isdone(self,iterator):
        return self.max_iter <= iterator.get_iteration()
    def __str__(self):
        return 'MaxIterTerminationCondition {0}'.format(self.max_iter)