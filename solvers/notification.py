import numpy as np
import math
   
#############################
# Notifications

class Notification(object):
    def announce(self,iterator):
        raise NotImplementedError()

class ValueChangeAnnounce(Notification):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self,**kwargs):
        self.old_v = np.NaN # Don't have old iteration
        self.diff = float('inf')
        
    def announce(self,iterator):
        v = iterator.get_value_vector()
        if np.any(self.old_v == np.NaN):
            self.old_v = v
            return False
            
        new_diff = np.linalg.norm(self.old_v - v)
        self.old_v = v
        
        if new_diff == 0 or math.log(new_diff) <= math.log(self.diff) - 1:
            print 'Value iteration diff {0:.3g} at iteration {1}'.format(new_diff,iterator.get_iteration())
            self.diff = new_diff

class ValueDiffAnnounce(Notification):
    """
    Prints the value difference
    """
    def __init__(self,**kwargs):
        self.old_v = np.NaN # Don't have old iteration
        
    def announce(self,iterator):
        v = iterator.get_value_vector()

        if not np.any(self.old_v == np.NaN):
            new_diff = np.linalg.norm(self.old_v - v)
            print 'Value iteration diff {0:.3g} at iteration {1}'\
                .format(new_diff,iterator.get_iteration())
        self.old_v = v        

class PrimalChangeAnnounce(Notification):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self,**kwargs):
        self.old_x = np.NaN # Don't have old iteration
        self.diff = float('inf')

        # Only use a slice of the primal vector
        self.indices = kwargs.get('indices',np.empty(0))
        
    def announce(self,iterator):
        x = iterator.get_primal_vector()
        if not self.indices:
            self.indices = slice(0,x.size)
        x = x[self.indices]
            
        if np.any(self.old_x == np.NaN):
            self.old_x = x
            
        new_diff = np.linalg.norm(self.old_x - x)
        self.old_x = x
        
        if new_diff == 0 or  math.log(new_diff) <= math.log(self.diff) - 1:
            print 'Primal iteration diff {0:.3g} at iteration {1}'.format(new_diff,iterator.get_iteration())
            self.diff = new_diff

class PrimalDiffAnnounce(Notification):
    """
    Reports iteration to iteration difference
    """
    def __init__(self,**kwargs):
        self.old_x = np.NaN # Don't have old iteration

        # Only use a slice of the primal vector
        self.indices = kwargs.get('indices',np.empty(0))
        
    def announce(self,iterator):
        x = iterator.get_primal_vector()
        if not self.indices:
            self.indices = slice(0,x.size)
        x = x[self.indices]
            
        if np.any(self.old_x == np.NaN):
            self.old_x = x
            return
            
        diff = np.linalg.norm(self.old_x - x)
        self.old_x = x
        
        print 'Primal iteration diff {0:.3g} at iteration {1}'\
            .format(diff,iterator.get_iteration())

class ResidualChangeAnnounce(Notification):
    """
    Checks if the value vector from an MDP iteration has changed 
    significantly
    """
    def __init__(self):
        self.residual_log = float('inf')
        
    def announce(self,iterator):
        x = iterator.get_primal_vector()
        w = iterator.get_dual_vector()

        new_r = np.linalg.norm(np.minimum(x,w))
        int_log_r = int(math.log(new_r))
        if int_log_r < self.residual_log:            
            print 'Residual {0:.3g} at iteration {1}'\
                .format(new_r,iterator.get_iteration())
            self.residual_log = int_log_r
            
class IterAnnounce(Notification):
    """
    Broadcast header every iteration
    """
    def __init__(self):
        pass
    def announce(self,iterator):
        print '-'*5,iterator.get_iteration(),'-'*5

class ResidualAnnounce(Notification):
    """
    Broadcast residual every iteration
    """
    def __init__(self):
        pass
    def announce(self,iterator):
        x = iterator.get_primal_vector()
        w = iterator.get_dual_vector()
        N = x.size
        
        r = np.linalg.norm(np.minimum(x,w))
        print 'Residual {0:.3g} at iteration {1}'\
            .format(r,iterator.get_iteration())

class PotentialAnnounce(Notification):
    """
    Broadcast potential every iteration
    """
    def __init__(self,k):
        self.k = k
        
    def announce(self,iterator):
        x = iterator.get_primal_vector()
        y = iterator.get_dual_vector()
        
        N = x.size
        k = self.k
              
        P = (N + k) * np.log(x.dot(y))\
            - np.sum(np.log(x)) - np.sum(np.log(y))
        
        print 'Potential {0:.3g} at iteration {1}'\
            .format(P,iterator.get_iteration())
        
        print x.dot(y),np.sum(x),np.sum(y)
