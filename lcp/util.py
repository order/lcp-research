import numpy as np
import math

def isvector(x,N=None):    
    if not len(x.shape) == 1:
        return False
        
def ismatrix(A,N=None,M=None):
    if not len(A.shape) == 2:
        return False
    (n,m) = A.shape
    if N and not n == N:
        return False
    if M and not m == M:
        return False
    return True
    
def issquare(A,N=None):
    return ismatrix(A,N,N)
    
def nonneg_proj(x):
    """
    Projections onto non-negative orthant; the []_+ operator
    """
    assert(isvector(x))
    return np.maximum(np.zeros(x.size),x)
    

def proj_forward_prop(x,M,q,w):
    """
    Does the projected version of Gauss-Siedel-ish forward solving
    """
    N = len(x)
    y = np.zeros(N)
    for i in xrange(N):
        curr_round_terms = M[i,0:i].dot(y[0:i])
        past_round_terms = M[i,i:].dot(x[i:])
        y[i] = max(0,x[i] - w * (1/M[i][i]) * (q[i] + curr_round_terms  + past_round_terms))
    return y
    
def basic_residual(x,w):
    """
    A basic residual for LCPs; will be zero if 
    """
    res = np.minimum(x,w)
    return np.linalg.norm(res)
    
# Fisher-Burmeister residual
def fb_residual(x,w):
    fb = np.sqrt(x**2 + w**2) - x - w
    return np.linalg.norm(fb)
    
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
        
############################
# Recording functions
class Recorder(object):
    def report(self,iteration):
        raise NotImplementedError()
        
class PrimalRecorder(Recorder):
    def __init__(self):
        pass
        
    def report(self,iteration):
        return iteration.get_primal_vector()
   
   
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
    def __init__(self):
        self.old_v = np.NaN # Don't have old iteration
        self.diff = float('inf')
        
    def announce(self,iterator):
        v = iterator.get_value_vector()
        if np.any(self.old_v == np.NaN):
            self.old_v = v
            return False
            
        new_diff = np.linalg.norm(self.old_v - v)
        self.old_v = v
        
        if math.log(new_diff) <= math.log(self.diff) - 1:
            print 'Value iteration diff {0:.3g} at iteration {1}'.format(new_diff,iterator.get_iteration())
            self.diff = new_diff
         
#############################
# Plotting for records

def image_matrix(A):
    f,ax = plt.subplots()
    ax.imshow(A, interpolation='nearest')
    plt.show()

def plot_state_img(record,**kwargs):
    """ Plots the state trajectory as an image
    """
    max_len = kwargs.get('max_len',2**64-1)    
    S = np.array(record.states)
    f,ax = plt.subplots()
    Img = (S - record.states[-1]) / (record.states[-1] + 1e-12) + 1e-12
    l = min(max_len,Img.shape[0])
    ax.imshow(np.log(np.abs(Img[:l,:].T)), interpolation='nearest')
    ax.set_title('Relative log change from final iter')
    plt.show()
    
