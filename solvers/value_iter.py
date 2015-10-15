from solvers import MDPIterator
from lcp import LCPObj

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
