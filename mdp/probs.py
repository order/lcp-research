import numpy as np

class ActionProbability(object):
    def get_prob(self,states):
        raise NotImplementedError()

    def get_single_prob(self,state):
        assert(1 == len(state.shape))
        P = self.get_prob(state[np.newaxis,:])[0,:]
        assert(1 == len(P.shape))
        return P

    def sample(self,states):
        raise NotImplementedError()

class UniformProbability(ActionProbability):
    def __init__(self,A):
        self.A = A
    def get_prob(self,states):
        (N,d) = states.shape
        return np.full((N,self.A),1.0 / float(self.A))
    
class FunctionProbability(ActionProbability):
    def __init__(self,fns):
        self.fns = fns

    def get_prob(self,states):
        (N,d) = states.shape
        A = len(self.fns)

        P = np.empty((N,A))

        for a in xrange(A):
            P[:,a] = self.fns[a].evaluate(states)
        assert(not np.any(P < 0))

        Z = np.sum(P,axis=1)
        assert((N,) == Z.shape)

        P = P / Z
        assert((N,A) == P.shape)
        return P
        
