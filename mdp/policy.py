import numpy as np

class Policy(object):
    """
    Abstract class for policies
    """
    def get_decisions(self,states):
        raise NotImplementedError()
        
class ConstantPolicy(Policy):
    """
    Constant dumb policy
    """
    def __init__(self,decision):
        self.decision = decision
        
    def get_decisions(self,states):
        m = len(states.shape) # Remap vectors to row vectors
        if 1 == m:
            states = states[np.newaxis,:] 
            
        (N,d) = states.shape
        return self.decision*np.ones(N)
        
class OneStepLookaheadPolicy(Policy):
    """
    Basic policy based on looking at the value for the next state for each action
    """
    def __init__(self, discretizer, value_fun_eval,discount):
    
        self.state_remapper = discretizer.physics
        self.value_fun_eval = value_fun_eval
        self.cost_obj = discretizer.cost_obj
        self.actions = discretizer.actions
        self.discount = discount
        
    def get_decisions(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:] 
            
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
    def __init__(self, discretizer, value_fun_eval, discount, k):
        self.state_remapper = discretizer.physics
        self.value_fun_eval = value_fun_eval
        self.cost_obj = discretizer.cost_obj
        self.actions = discretizer.actions
        self.discount = discount
        self.k = k
        assert(type(k) == int)

        
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
    
