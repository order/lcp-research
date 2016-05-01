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
