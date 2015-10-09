import numpy as np
import scipy
import scipy.sparse as sps
import matplotlib.pyplot as plt
import lcp
import scipy
import pickles

class MDP(object):
    """
    MDP object
    """
    def __init__(self,*vargs,**kwargs):
        if (1 == len(vargs)):
            # Load from file
            filename = vargs[0]
            loader = scipy.load(filename)
            transitions = pickles.pickle_array_to_multi_matrix(loader['transitions'])
            long_costs = loader['costs']
            actions = loader['actions']
            
            # Split up the costs from one long array to A small ones
            N = transitions[0].shape[0]
            A = len(actions)
            assert(long_costs.size == N*A)
            costs = [long_costs[(N*i):(N*(i+1))] for i in xrange(A)]
            
            self.name = kwargs.get('name',loader['name'])
            self.discount = kwargs.get('discount',loader['discount'])
            
        elif (3 == len(vargs)):
            # Get directly from command line
            (transitions,costs,actions) = vargs
            self.discount = kwargs.get('discount',0.99)
            self.name = kwargs.get('name','Unnamed')
        else:
            assert(len(vargs) in [1,3])            
            
        self.transitions = transitions
        self.costs = costs
        self.actions = actions
        
        A = len(actions)
        N = costs[0].size
        self.num_actions = A
        self.num_states = N        

        assert(len(transitions) == A)
        assert(len(costs) == A)
        
        # Ensure sizes are consistent
        for i in xrange(A):
            assert((N,) == costs[i].shape)
            assert(not np.any(np.isnan(costs[i])))
            
            assert((N,N) == transitions[i].shape)
            assert(abs(transitions[i].sum() - N)/N <= 1e-12)
            
    def write(self,filename):
        transition_array = pickles.multi_matrix_to_pickle_array(self.transitions)
        long_costs = np.concatenate(self.costs)
        scipy.savez(filename,\
            transitions=transition_array,\
            costs=long_costs,\
            actions=self.actions,\
            name=self.name,\
            discount=self.discount)
        
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        
        I guess we're using a sparse matrix here...
        """
        return sps.eye(self.num_states,format='lil') - self.discount * self.transitions[a].T

    def tolcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        #Build the LCP
        Top = sps.lil_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -np.ones(n)
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = sps.hstack((-E,sps.coo_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = sps.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = sps.hstack((Top,E.T))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = sps.vstack((Top,Bottom))
        
        return lcp.LCPObj(M,q,name='LCP from {0} MDP'.format(self.name))

    def __str__(self):
        return '<{0} with {1} actions and {2} states>'.\
            format(self.name, self.num_actions, self.num_states)
        
        
class MDPValueIterSplitter(object):
    """
    Builds an LCP based on an MDP, and split it
    according to value-iteration based (B,C)
    """
    def __init__(self,MDP,**kwargs):
        self.MDP = MDP
        self.num_actions = self.MDP.num_actions
        self.num_states = self.MDP.num_states
        
        # Builds an LCP based on an MPD
        self.LCP = lcp.MDPLCPObj(MDP)
        self.value_iter_split()      
    
    def update(self,v):
        """
        Builds a q-vector based on current v
        """
        q_k = self.LCP.q + self.C.dot(v)
        return (self.B,q_k)    

    def value_iter_split(self):
        """
        Creates a simple LCP based on value-iteration splitting:
        M = [[0 I I], [-I 0 0], [-I 0 0]]
        """
        I_list = []
        P_list = []
        # Build the B matrix
        for i in xrange(self.num_actions):
            I_list.append(sps.eye(self.num_states))
        self.B = mdp_skew_assembler(I_list)
        self.C = self.LCP.M - self.B

def mdp_skew_assembler(A_list):
    """
    Builds a skew-symmetric block matrix from a list of squares
    """
    A = len(A_list)
    (n,m) = A_list[0].shape
    assert(n == m) # Square
    N = (A+1)*n
    M = sps.lil_matrix((N,N))
    for i in xrange(A):
        I = xrange(n)
        J = xrange((i+1)*n,(i+2)*n)
        M[np.ix_(I,J)] = A_list[i]
        M[np.ix_(J,I)] = -A_list[i]
    return M.tocsr()
        