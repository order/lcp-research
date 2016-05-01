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
    def __init__(self):
        self.actions = np.array([[-1],[0],[1]])
        
    def get_decisions(self,points):
        I = self.get_decision_indices(points)
        return self.actions[I,:]

    def get_decision_indices(self,points):
        (N,d) = points.shape
        assert(d == 2)

        v = points[:,1]
        x = points[:,0]
        
        mask = v + np.sign(x) * np.sqrt(2 * np.abs(x)) > 0

        I = 2*np.ones(N,dtype='int')
        I[mask] = 0

        return I

    def get_action_dim(self):
        return 1   
