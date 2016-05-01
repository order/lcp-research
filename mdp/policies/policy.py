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
    def get_action_dim(self):
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
    def get_action_dim(self):
        return self.A
    
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
        

    
