    
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
    def isdone(self,iteration):
        raise NotImplementedError()

class MaxIterTerminationCondition(TerminationCondition):
    def __init__(self,max_iter):
        self.max_iter = max_iter
    def isdone(self,iteration):
        return self.max_iter <= iteration.get_iteration()
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
    
