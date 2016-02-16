import numpy as np

class Problem(object):
    """
    A problem is essentially the continuous part of a physics problem.
    It contains the physics, the boundary description, costs
    """
    def __init__(self,
                 physics,
                 boundary,
                 cost_obj,
                 weight_obj,
                 action_dim,
                 discount):
        
        
        self.physics = physics # How physical states map to others
        self.boundary = boundary # [(low,high),(low,high),...]
        self.exception_state_remappers = []
        self.dimension = len(boundary)

        self.costs = cost_obj
        self.weights = weight_obj

        self.action_dim = action_dim
        
        if (1 == len(actions.shape)):
            actions = actions[:,np.newaxis] # convert to column vector
            
        self.discount = discount
        
    def get_num_actions(self):
        """
        Return the number of actions
        """
        return self.num_actions
        
    def get_boundary(self):
        """
        Returns a list of pairs, with the min and max along each dimension.
        So [(-1,1),(-5,5)] could be the boundary for a problem, indicating
        that the problem is 2D with a rectangular geometry [-1,1] x [-5,5]
        """
        return self.boundaries

    def get_dimension(self):
        """
        Return the number of physical dimensions in the problem,
        e.g. 2 for the 1D double integrator (position and velocity)
        """
        return self.dimension

    def next_states(self,states,actions,**kwargs)):
        """
        Returns a successor state for every state.

        states - (N,d) array of states
        actions - (N,u) array of actions
        """

        assert(2 == len(states.shape))
        (N,d) = states.shape
        assert(d == self.dimension)
        U = self.action_dim

        
        if 1 == len(actions.shape):
            uniform = True
            assert((U,) == actions.shape))
        else:
            uniform = False
            assert((N,U) == actions.shape)


        # Initial map
        if uniform:
            next_states = self.physics.remap(states,action=action)
        else:
            next_states = np.empty((N,d))
            for i in xrange(N):
                next_states[i,:] = self.physics.remap(states[i,:],
                                                      action=actions[i,:])
        # Enforce boundaries
        for remapper in self.exception_state_remappers:
            next_states = remapper.remap(next_states)
            
        return next_states

    def out_of_bounds(self,points):
        """
        Check if points are out of bounds for the rectangular boundaries
        """
        L = np.array([low for (low, high) in self.boundaries])
        U = np.array([high for (low,high) in self.boundaries]])
        return np.any(points < L,axis=1) + np.any(points < U,axis=1)

