import numpy as np

from solvers import MDPIterator
from utils.parsers import KwargParser
import utils

class TabularTDIterator(MDPIterator):
    """
    TD(0) scheme based on an MDP;
    Want to remove the use of a model in a subsequent
    variant.
    """
    
    def __init__(self,mdp_obj,**kwargs):
        parser = KwargParser()
        parser.add('num_samples')
        parser.add('step_size')
        parser.add('policy')
        args = parser.parse(kwargs)
        self.__dict__.update(args)
        
        self.mdp_obj = mdp_obj
        N = mdp_obj.num_states
        assert((N,) == self.policy.shape)

        self.v = np.zeros(N) # Assume tabular form
        self.iteration = 0

    def next_iteration(self):
        # 1) Sample S
        # 2) Get action from policy
        # 3) Get successor state S' and cost C
        # 4) V(S) <- V(S) + alpha[C + gamma V(S') - V(S)]
        
        alpha = self.step_size
        gamma = self.mdp_obj.discount

        mdp_obj = self.mdp_obj
        V = self.v

        N = mdp_obj.num_states
        A = mdp_obj.num_actions
        
        S = np.random.randint(0,N,self.num_samples) # Uniform indices
        actions = self.policy[S]

        for a in xrange(A):
            a_mask = (actions == a) # Mask for action = a
            n = a_mask.sum()
            if 0 == n:
                continue
            
            SA = S[a_mask] # State/action indicies
            assert((n,) == SA.shape)
            
            costs = mdp_obj.costs[a][SA]
            assert((n,) == costs.shape)

            # Using the whole column
            S_next = mdp_obj.transitions[a][:,SA]
            assert((N,n) == S_next.shape)

            V_next = (S_next.T).dot(V)
            assert((n,) == V_next.shape)

            V[SA] = (1.0 - alpha)*V[SA] + alpha * (costs + gamma * V_next)

        self.iteration += 1        

    def get_value_vector(self):
        return np.array(self.v)
        
    def get_iteration(self):
        return self.iteration
