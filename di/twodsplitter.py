import numpy as np

class TwoDSplitter(object):
    """
    Splits result of solve into a series of 2D image matrices
    """
    def __init__(self,A,n,x,y):
        """
        Get all the dimensionality information
        """
        self.num_actions = A
        self.num_states = n
        assert(x*y <= n)
        self.x_dim = x
        self.y_dim = y

    def split(self,X):
        """
        Split a I x N primal/dual MDP matrix into 2D matrices
        """
        A = self.num_actions
        n = self.num_states
        x = self.x_dim
        y = self.y_dim

        if 1 == len(X.shape):
            X = X[np.newaxis,:]
            
        (I,N) = X.shape
        assert(N == (A+1)*n)

        Splits = []
        for i in xrange(I):
            Imgs = []
            for a in xrange(A+1):
                sl = slice(a*n,a*n + x*y)
                print sl
                vect = X[i,sl]
                assert((x*y,) == vect.shape)
                img = np.reshape(vect,(x,y))
                Imgs.append(img)
            Splits.append(Imgs)

        return Splits
