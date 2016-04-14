import numpy as np

class Policy(object):
    def get_decisions(self,points):
        raise NotImplementedError()
    def get_single_decision(self,point):
        return self.get_decisions(point[np.newaxis,:])[0,:]
    def get_action_dim(self):
        raise NotImplementedError()
        

class IndexPolicy(object):
    def get_decision_indices(self,points):
        raise NotImplementedError()
    def get_single_decision_index(self,point):
        return self.get_decision_indices(point[np.newaxis,:])[0]

class IndexPolicyWrapper(Policy):
    """
    Converts an index policy into a normal policy
    """
    def __init__(self,policy,actions):
        self.policy = policy
        self.actions = actions
        (A,dimA) = actions.shape
        self.num_actions = A
        self.action_dim = dimA
    def get_decisions(self,points):
        aid = self.policy.get_decision_indices(points)
        return self.actions[aid,:]
    def get_action_did(self):
        return self.action_dim        
    
class ConstantDiscretePolicy(IndexPolicy):
    def __init__(self,action_index):
        self.action_index = action_index
    def get_decision_indices(self,points):
        N = points.shape[0]
        actions = np.full((N,1),self.action_index)
        return actions

class UniformDiscretePolicy(IndexPolicy):
    def __init__(self,num_actions):
        self.A = num_actions
    def get_decision_indices(self,points):
        N = points.shape[0]
        return np.random.randint(self.A,size=(N,1))

class RandomDiscretePolicy(IndexPolicy):
    def __init__(self,p):
        self.p = p
    def get_decision_indices(self,points):
        N = points.shape[0]
        (A,) = self.p.shape
        return np.random.choice(A,size=N,p=self.p)

class ConstantPolicy(Policy):
    def __init__(self,action):
        assert(1 == len(action.shape))
        self.action = action
    def get_decisions(self,points):
        (N,d) = points.shape
        (u,) = self.action.shape
        decision = np.tile(self.action,(N,1))
        assert((N,u) == decision.shape)
        return decision
    def get_action_dim(self):
        return self.action.size

class EpsilonFuzzedPolicy(IndexPolicy):
    def __init__(self,num_actions,epsilon,policy):
        self.A = num_actions
        self.e = epsilon
        self.policy = policy
    def get_decision_indices(self,points):
        (N,d) = points.shape
        decision = self.policy.get_decision_indices(points)
        (M,) = decision.shape
        assert(N == M)
        
        mask = (np.random.rand(N) < self.e)
        S = np.sum(mask)
        rand_aid = np.random.randint(self.A,size=S)
        
        decision[mask] = rand_aid
        return decision

class MinFunPolicy(IndexPolicy):
    def __init__(self,fns):
        self.fns = fns
        
    def get_decision_indices(self,points):
        (N,d) = points.shape
        A = len(self.fns)
               
        F = np.empty((N,A))
        for a in xrange(A):
            F[:,a] = self.fns[a].evaluate(points)
        return np.argmin(F,axis=1)
    
class MaxFunPolicy(IndexPolicy):
    def __init__(self,fns):
        self.fns = fns
        
    def get_decision_indices(self,points):
        (N,d) = points.shape
        A = len(self.fns)
               
        F = np.empty((N,A))
        for a in xrange(A):
            F[:,a] = self.fns[a].evaluate(points)
        return np.argmax(F,axis=1)

class SoftMaxFunPolicy(IndexPolicy):
    def __init__(self,fns):
        self.fns = fns
        
    def get_decision_indices(self,points):
        (N,d) = points.shape
        A = len(self.fns)
               
        F = np.empty((N,A))
        for a in xrange(A):
            F[:,a] = self.fns[a].evaluate(points)

        F = F - np.max(F)
        Z = np.sum(F,axis=1)
        P = F / Z
        decision = np.empty(N)
        for i in xrange(N):
            decision[i] = np.random.choice(A,p=P[i,:])
        return decision
        
class KStepLookaheadPolicy(Policy):
    """
    Basic policy based on looking at the value for the next state for each action
    """
    def __init__(self, discretizer, value_fun_eval, discount, k):
        self.discretizer = discretizer
        self.state_remapper = discretizer.physics
        self.value_fun_eval = value_fun_eval
        self.cost_obj = discretizer.cost_obj
        self.actions = discretizer.actions
        self.discount = discount
        self.k = k
        assert(type(k) == int)

        
    def get_values(self,states,k):
        """
        Calculates the Q-values from the value function:
        Q(s,a) = C(s,a) + \gamma * \sum_{s'} T(s',a,s) * V(s')
        """
        (N,d) = states.shape
        assert(d == self.discretizer.basic_mapper.get_dimension())
        (A,ad) = self.actions.shape
        
        # Base case; vals are just the value function
        if k <= 0:
            # Kludge: shouldn't actually bother duplicating
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
        (A,ad) = self.actions.shape
        assert(d == self.discretizer.basic_mapper.get_dimension())
    
        # Recursively get the values
        vals = self.__get_vals(states,self.k) 
        
        # Take the arg min 
        action_indices = np.argmin(vals,axis=1)
        assert(action_indices.shape[0] == N)
        
        # Convert indices to actions.
        decisions = np.empty((N,ad))
        for i in xrange(A):
            mask = (action_indices == i)
            decisions[mask,:] = self.actions[i,:]         
        assert(not np.any(np.isnan(decisions)))
        
        return decisions
        
class LinearFeedbackPolicy(Policy):
    """
    Linear feedback gain policy; think LQR
    """
    def __init__(self,gains,offset,**kwargs):
        (n,m) = gains.shape
        assert((m,) == offset.shape)
        
        self.K = gains
        self.offset = offset 
        self.lim = kwargs.get('limit',(-float('inf'),float('inf')))
        
    def get_decisions(self,states):
        m = len(states.shape) # Remap vectors to row vectors
        if 1 == m:
            states = states[np.newaxis,:] 
        
        (NumPoints,NumDim) = states.shape
        (NumActions,D) = self.K.shape
        assert(NumDim == D)
        
        decisions = -self.K.dot(states.T - self.offset[:,np.newaxis]).T # u = -Kx
        decisions = np.maximum(self.lim[0],np.minimum(self.lim[1],decisions))
        assert((NumPoints,NumActions) == decisions.shape)
        
        return decisions 
    
