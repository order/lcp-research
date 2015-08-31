import numpy as np
import scipy
import matplotlib.pyplot as plt
import lcp

class MDP(object):
    """
    MDP object
    """
    def __init__(self,transitions,costs,actions,**kwargs):
        self.discount = kwargs.get('discount',0.99)
        self.transitions = transitions
        A = len(actions)
        N = costs[0].shape[0]
        self.actions = actions
        self.name = kwargs.get('name','Unnamed')
        self.num_actions = A
        self.num_states = N
        
        self.costs = costs

        assert(len(transitions) == A)
        assert(len(costs) == A)
        
        # Ensure sizes are consistent
        for i in xrange(A):
            assert(costs[i].size == N)
            assert(not np.any(np.isnan(costs[i])))
            
            assert(transitions[i].shape[0] == N)
            assert(transitions[i].shape[1] == N)
            assert(abs(transitions[i].sum() - N) <= 1e-6)
            
        
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        
        I guess we're using a sparse matrix here...
        """
        return np.eye(self.num_states) - self.discount * self.transitions[a].T

    def tolcp(self):
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        Top = scipy.sparse.coo_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -np.ones(n)
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = scipy.sparse.hstack((-E,scipy.sparse.coo_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = scipy.sparse.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = scipy.sparse.hstack((Top,E.T))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = scipy.sparse.vstack((Top,Bottom))
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
            I_list.append(scipy.sparse.eye(self.num_states))
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
    M = scipy.sparse.lil_matrix((N,N))
    for i in xrange(A):
        I = xrange(n)
        J = xrange((i+1)*n,(i+2)*n)
        M[np.ix_(I,J)] = A_list[i]
        M[np.ix_(J,I)] = -A_list[i]
    return M.tocsr()
    
class ValueFunctionEvaluator(object):
    def evaluate(self,state,action):
        raise NotImplementedError()
        
class InterpolatedGridValueFunctionEvaluator(ValueFunctionEvaluator):
    def __init__(self,discretizer,v):
        self.node_to_cost = v
        self.discretizer = discretizer
        
    def evaluate(self,states):
        # Convert state into node dist
        node_dists = self.discretizer.states_to_node_dists(states)        
        
        vals = np.zeros(states.shape[0])
        for (state_id,nd) in node_dists.items():
            for (node_id,w) in nd.items():
                vals[state_id] += self.node_to_cost[node_id] * w                
        return vals
        
class Policy(object):
    """
    Abstract class for policies
    """
    def get_decisions(self,states):
        raise NotImplementedError()
        
class OneStepLookaheadPolicy(Policy):
    """
    Basic policy based on looking at the value for the next state for each action
    """
    def __init__(self, cost_obj, state_remapper, value_fun_eval, actions,discount):
        self.state_remapper = state_remapper
        self.value_fun_eval = value_fun_eval
        self.cost_obj = cost_obj
        self.actions = actions
        self.discount = discount
        
    def get_decisions(self,states):
        (N,d) = states.shape
        A = len(self.actions)
                    
        # Get the values for these states
        vals = np.full((N,A),np.nan)
        for (i,a) in enumerate(self.actions):
            next_states = self.state_remapper.remap(states,action=a)
            vals[:,i] = self.cost_obj.cost(states,a) \
                + self.discount*self.value_fun_eval.evaluate(next_states)
        
        # Take the arg min, convert indices to actions.
        action_indices = np.argmin(vals,axis=1)
        assert(action_indices.shape[0] == N)
        decisions = np.full(N,np.nan)
        for (i,a) in enumerate(self.actions):
            mask = (action_indices == i)
            decisions[mask] = a
            
        assert(not np.any(np.isnan(decisions)))
        
        return decisions
        
class KStepLookaheadPolicy(Policy):
    """
    Basic policy based on looking at the value for the next state for each action
    """
    def __init__(self, cost_obj, state_remapper, value_fun_eval, actions, discount, k):
        self.state_remapper = state_remapper
        self.value_fun_eval = value_fun_eval
        self.cost_obj = cost_obj
        self.actions = actions
        self.k = k
        assert(type(k) == int)
        self.discount = discount

        
    def __get_vals(self,states,k):
        (N,d) = states.shape
        A = len(self.actions)        
        # Base case; vals are just the value function
        if k <= 0:
            # Kludge: don't bother acutally duplicating
            vals = np.tile(self.value_fun_eval.evaluate(states),(1,1)).T
            return vals

        vals = np.empty((N,A))
        for (i,a) in enumerate(self.actions):
            # Get the next states for each action from the physics
            next_states = self.state_remapper.remap(states,action=a)            
            # Value is immediate costs, plus best for best cost-to-go for next state
            vals[:,i] = self.cost_obj.cost(states,a)\
                + self.discount*np.amin(self.__get_vals(next_states,k-1),axis=1)                
        return vals
        
    def get_decisions(self,states):
        (N,d) = states.shape
    
        # Recursively get the values
        vals = self.__get_vals(states,self.k) 
        
        # Take the arg min 
        action_indices = np.argmin(vals,axis=1)
        assert(action_indices.shape[0] == N)
        
        # Convert indices to actions.
        decisions = np.empty(N)
        for (i,a) in enumerate(self.actions):
            mask = (action_indices == i)
            decisions[mask] = a           
        assert(not np.any(np.isnan(decisions)))
        
        return decisions
    
