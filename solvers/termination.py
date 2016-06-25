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

class PrimalChangeTerminationCondition(TerminationCondition):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self,thresh):
        self.thresh = thresh
        self.old_x = np.NaN # Don't have old iteration
        self.diff = float('inf')
        
    def isdone(self,iterator):
        x = iterator.get_primal_vector()
        if np.any(self.old_x == np.NaN):
            self.old_x = x
            return False
            
        self.diff = np.linalg.norm(self.old_x - x)
        self.old_x = x
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
        N = x.size
            
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

class PotentialTerminationCondition(TerminationCondition):
    """
    Broadcast potential every iteration
    """
    def __init__(self,thresh):
        self.thresh = thresh
        
    def isdone(self,iterator):
        x = iterator.get_primal_vector()
        w = iterator.get_dual_vector()
        
        N = x.size
        P = (N + np.sqrt(N)) * np.log(x.dot(w)) \
            - np.sum(np.log(x)) - np.sum(np.log(w))

        return P <= self.thresh

    def __str__(self):
        return 'PotentialTerminationCondition {0}'.format(self.thresh)
        
class PotentialDiffTerminationCondition(TerminationCondition):
    """
    Broadcast potential every iteration
    """
    def __init__(self,thresh):
        self.thresh = thresh
        self.old_P = np.nan
        
    def isdone(self,iterator):
        p = iterator.get_primal_vector()
        d = iterator.get_dual_vector()
        
        N = p.size
        P = (N + np.sqrt(N)) * np.log(p.dot(d)) \
            - np.sum(np.log(p)) - np.sum(np.log(d))

        diff = np.abs(P - self.old_P)
        self.old_P = P

        return diff <= self.thresh

    def __str__(self):
        return 'PotentialTerminationCondition {0}'.format(self.thresh)
