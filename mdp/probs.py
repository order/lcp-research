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
        return np.full((N,self.A),1.0 / float(self.A),dtype=np.double)
    
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
        assert(not np.any(Z < 0))
        mask = (Z == 0)
        n_mask = mask.sum()
        
        if n_mask < N:
            P[~mask,:] = P[~mask,:] / Z[~mask]
        if n_mask > 0: 
            P[mask,:] = 1.0 / float(A)
        assert (np.abs(np.sum(P) - N) / float(N) < 1e-15)
        assert((N,A) == P.shape)
        return P
        
