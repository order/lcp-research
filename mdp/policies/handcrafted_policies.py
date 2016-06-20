import numpy as np
from policy import Policy,IndexPolicy

class HallwayPolicy(Policy,IndexPolicy):
    def __init__(self,N):
        self.N = N
        self.target = int(N/2)
        self.actions = np.array([[-1],[0],[1]])
        
    def get_decisions(self,points):
        I = self.get_decision_indices(points)
        return self.actions[I,:]

    def get_decision_indices(self,points):
        (N,d) = points.shape
        assert(d == 1)

        x = points[:,0]
        I = np.empty(N,dtype=np.int)
        I[x < self.target] = 2
        I[x == self.target] = 1
        I[x > self.target] = 0
        return I

    def get_action_dim(self):
        return 1      

class BangBangPolicy(Policy,IndexPolicy):
    def __init__(self,actions):
        (A,ad) = actions.shape
        assert(actions[0] == -actions[-1])
        assert(1 == ad)

        self.actions = actions
        
    def get_decisions(self,points):
        I = self.get_decision_indices(points)
        return self.actions[I,:]

    def get_decision_indices(self,points):
        (N,d) = points.shape
        (A,ad) = self.actions.shape
        assert(d == 2)

        v = points[:,1]
        x = points[:,0]
        
        mask = v + np.sign(x) * np.sqrt(2 * np.abs(x)) > 0
        I = (A-1)*self.actions.shapnp.ones(N,dtype='int')
        I[mask] = 0

        return I

    def get_action_dim(self):
        return 1
    
class LinearPolicy(Policy,IndexPolicy):
    """
    Decision boundary defined by <x,a> = 0

    If <x,a> >= 0 then do action a[0]
    O.w. do action a[-1]
    """
    def __init__(self,actions,a):
        self.actions = np.array([[-1],[0],[1]])
        self.a = a
        
    def get_decisions(self,points):
        I = self.get_decision_indices(points)
        return self.actions[I,:]

    def get_decision_indices(self,points):
        (N,d) = points.shape
        (A,ad) = self.actions.shape
        (M,) = self.a.shape
        
        assert(M == d)

        mask = points.dot(self.a) >= 0

        I = (A-1)*np.ones(N,dtype='int')
        I[mask] = 0

        return I

    def get_action_dim(self):
        return 1   

class HillcarPolicy(Policy,IndexPolicy):
    def __init__(self,actions):
        self.actions = actions
        
    def get_decisions(self,points):
        I = self.get_decision_indices(points)
        return self.actions[I,:]

    def get_decision_indices(self,points):
        # Assuming we're using simple_slope
        pass

    def get_action_dim(self):
        return 1   
