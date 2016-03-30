import numpy as np
import scipy as sp
import scipy.sparse as sps

import mdp
import lcp
from utils.parsers import KwargParser

class TabularMDP(mdp.DiscreteMDP, lcp.LCPBuilder):
    """
    MDP object assuming a discrete state-space with tabular representation.
    """
    def __init__(self,transitions,
                 costs,
                 actions,
                 discount,
                 state_weights,
                 **kwargs):
        self.transitions = transitions
        self.costs = costs
        self.actions = actions
        self.discount = discount
        self.state_weights = state_weights

        parser = KwargParser()
        parser.add('name','Unnamed')
        args = parser.parse(kwargs)        
        self.name = args['name']

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
            # Stochastic checking removed
            
    def get_action_matrix(self,a):
        """
        Build the action matrix E_a = I - \gamma * P_a^\top 
        
        I guess we're using a sparse matrix here...
        """
        return sps.eye(self.num_states,format='lil')\
            - self.discount * self.transitions[a]

    def next_state_index(self,state,action):
        """
        Takes in state and action INDEXES (already discretized)
        and returns a reward and sampled next state
        """
        
        N = self.num_states
        A = self.num_actions
        assert(0 <= state <= N)
        assert(0 <= action <= A)

        # Get the reward for doing action in state
        reward = self.costs[action][states]

        T = self.transitions[action]
        if isinstance(T,np.ndarray):
            # Dense sampling
            dist = T[:,state]
            assert((N,) == dist.shape)
            next_state = np.random.choice(xrange(N),p=dist)
        else:
            # Sparse sampling
            assert(isinstance(T,sps.spmatrix))
            dist = (T.tocoo()).getcol(state)
            next_state = np.random.choice(dist.row,p=dist.data)

        return (next_state,reward)        

    def get_value_residual(self,v):
        N = self.num_states
        A = self.num_actions
        
        comps = np.empty((N,A))
        gamma = self.discount
        for a in xrange(A):
            c = self.costs[a]
            Pt = self.transitions[a].T
            comps[:,a] = c + gamma * Pt.dot(v)

        res = v - np.amin(comps,axis=1)
        assert((N,) == res.shape)
        return res

    def build_lcp(self,**kwargs):
        # Optional regularization
        parser = KwargParser()
        parser.add('value_regularization',0.0)
        parser.add('flow_regularization',0.0)
        args = parser.parse(kwargs)        
        self.val_reg = args['value_regularization']
        self.flow_reg = args['flow_regularization']       
        
        n = self.num_states
        A = self.num_actions
        N = (A + 1) * n
        d = self.discount

        # Build the LCP
        Top = sps.lil_matrix((n,n))
        Bottom = None
        q = np.zeros(N)
        q[0:n] = -self.state_weights
        for a in xrange(self.num_actions):
            E = self.get_action_matrix(a)
            
            # NewRow = [-E_a 0 ... 0]
            NewRow = sps.hstack((-E.T,sps.lil_matrix((n,A*n))))
            if Bottom == None:
                Bottom = NewRow
            else:
                Bottom = sps.vstack((Bottom,NewRow))
            # Top = [...E_a^\top]
            Top = sps.hstack((Top,E))
            q[((a+1)*n):((a+2)*n)] = self.costs[a]
        M = sps.vstack((Top,Bottom))

        Reg = sps.lil_matrix((N,N))
        Reg[n:,n:] = self.flow_reg*sps.eye(A*n)
        Reg[:n,:n] = self.val_reg*sps.eye(n)
        
        M = M + Reg

        assert((N,N) == M.shape)
        assert((N,) == q.shape)
        
        return lcp.LCPObj(M,q,name='LCP from {0} MDP'.format(self.name))

    def __str__(self):
        return '<{0} with {1} actions and {2} states>'.\
            format(self.name, self.num_actions, self.num_states)
