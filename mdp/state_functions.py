import numpy as np

class RealFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class MultiFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class Basis(object):
    def get_basis(self,points):
        raise NotImplementedError()
    def get_orth_basis(self,points):
        raise NotImplementedError()
    
class TrigBasis(Basis):
    def __init__(self,freq,shift):
        (M,D) = freq.shape
        (m,) = shift.shape
        assert(m==M)        

        self.offset = np.zeros(D)
        self.freq = freq.T
        self.shift = shift

    def rescale(self,grid_desc):
        # Assume frequencies are scaled and shifted
        # to be on [0,1]*D
        # Modify so that they're periodic over the grid
        (D,M) = self.freq.shape
        assert(D == len(grid_desc))

        
        self.offset = np.empty(D)        
        for (d,(l,u,n)) in enumerate(grid_desc):
            # Shift so l is the 0
            self.offset[d] = l

            # Points run in (l,u) over N cells
            # There are N+1 distinct points in these cells
            # Therefore, we're periodic so that:
            # l = [u + (u - l) / n]
            # So the period is (u - l) + (u-l)/n
            period = (1.0 + 1.0 / float(n))*(u - l)
            self.freq[d,:] /= period

        
    def get_basis(self,points,**kwargs):
        (N,D) = points.shape
        (d,M) = self.freq.shape
        assert(D == d)

        # Shift points
        P = points - self.offset[np.newaxis,:]
        B = np.sin((P).dot(self.freq) + self.shift[np.newaxis,:])
        assert((N,M) == B.shape)
        return B

    def get_orth_basis(self,points):
        B = self.get_basis(points)

        # Eliminate dependant columns (multi-dimensional aliasing...)
        [Q,R] = np.linalg.qr(B)
        indep = (np.abs(np.diag(R)) > 1e-8)
        B = B[:,indep]

        # Normalize
        for i in xrange(B.shape[1]):
            B[:,i] /= np.linalg.norm(B[:,i])

        return B
    
class TrigFunction(MultiFunction):
    def __init__(self,freq,shift,amps):
        (M,D) = freq.shape
        (m,) = amps.shape
        assert(M == m)
        
        self.basis = TrigBasis(freq,shift)
        self.amps = amps

    def evaluate(self,points):
        B = self.basis.get_basis(points)
        (N,M) = B.shape
        (m,) = self.amps.shape
        assert(M==m)
        
        R = B.dot(self.amps)
        assert((N,) == R.shape)
        return R

    
class TrigMultiFunction(MultiFunction):
    def __init__(self,freq,shift,amps):
        (M,D) = freq.shape
        (m,K) = amps.shape
        assert(M == m)
        
        self.basis = TrigBasis(freq,shift)
        self.amps = amps

    def evaluate(self,points):
        B = self.basis.get_basis(points)
        (N,M) = B.shape
        (m,K) = self.amps.shape
        assert(M==m)
        
        R = B.dot(self.amps)
        assert((N,K) == R.shape)
        return R
        
    
class InterpolatedFunction(RealFunction):
    """
    Take in a finite vector and a discretizer;
    Enable evaluation at arbitrary points via interpolation
    """
    def __init__(self,discretizer,y):
        N = discretizer.num_nodes()
        assert((N,) == y.shape)
        self.target_values = y
        self.discretizer = discretizer
        
    def evaluate(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]

        y = self.target_values
        (N,) = y.shape
        (M,d) = states.shape
        
        # Convert state into node dist
        P = self.discretizer.points_to_index_distributions(states)
        assert((N,M) == P.shape)
        f = (P.T).dot(y)
        assert((M,) == f.shape)
        return f

class InterpolatedMultiFunction(MultiFunction):
    """
    Take in a finite vector and a discretizer;
    Enable evaluation at arbitrary points via interpolation
    """
    def __init__(self,discretizer,y):
        self.target_values = y
        self.discretizer = discretizer
        
    def evaluate(self,states):
        if 1 == len(states.shape):
            states = states[np.newaxis,:]

        y = self.target_values
        (N,A) = y.shape
        (M,d) = states.shape
        
        # Convert state into node dist
        P = self.discretizer.points_to_index_distributions(states)
        assert((N,M) == P.shape)
        f = (P.T).dot(y)
        assert((M,A) == f.shape)
        return f

class ConstFn(RealFunction):
    def __init__(self,v):
        self.v = v
    def evaluate(self,points):
        (N,D) = points.shape 
        return self.v * np.ones(N)

class FixedVectorFn(RealFunction):
    def __init__(self,x):
        assert(1 == len(x.shape))
        self.x = x
    def evaluate(self,points):
        (N,D) = points.shape 

        assert((N,) == self.x.shape)
        return self.x    

class GaussianFn(RealFunction):
    def __init__(self,bandwidth,
                 setpoint,
                 non_physical):
        
        assert(1 == len(setpoint.shape))
        N = setpoint.size
        self.setpoint = np.reshape(setpoint,(1,N))
        self.bandwidth = bandwidth
        self.non_phys = non_physical
 
    def evaluate(self,points):
        (N,D) = points.shape 
        
        norm = np.sum(np.power(points - self.setpoint,2),axis=1)
        costs = np.exp(-norm / self.bandwidth)
        assert((N,) == costs.shape)

        # Deal with overrides
        mask = np.isnan(costs)
        costs[mask] = self.non_phys
        
        return costs
        
class BallSetFn(RealFunction):

    def __init__(self,center,radius):
        
        self.center = center
        self.radius = radius
    
    def evaluate(self,points):       
        (N,d) = points.shape
        nan_mask = ~np.any(np.isnan(points),axis = 1)
        assert((N,) == nan_mask.shape)
        n = nan_mask.sum()

        inbound = np.zeros(N,dtype=bool)
        diff = points[nan_mask,:] - self.center
        assert((n,d) == diff.shape)
        r = np.linalg.norm(diff, axis=1)
        assert((n,) == r.shape)
        inbound[nan_mask] = (r <= self.radius)
        
        costs = np.empty(points.shape[0])
        costs[inbound] = 0.0
        costs[~inbound] = 1.0
        
        return costs
        
class TargetZoneFn(RealFunction):

    def __init__(self,targets):
        (D,_) = targets.shape
        self.targets = targets
        self.D = D
    
    def evaluate(self,points):
        (N,D) = points.shape
        assert(self.D == D)

        mask = np.any(np.isnan(points),axis=1)
        mask[~mask] = np.any(points[~mask,:] < self.targets[:,0],
                             axis=1)
        mask[~mask] = np.any(points[~mask,:] > self.targets[:,1],
                             axis=1)        

        costs = np.zeros(N)
        costs[mask] = 1.0

        return costs
