import numpy as np
import scipy.fftpack as fft
from utils.parsers import KwargParser

class RealFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class MultiFunction(object):
    def evaluate(self,points,**kwargs):
        raise NotImplementedError()

class Basis(object):
    def evaluate(self,points):
        raise NotImplementedError()


def top_trig_features(f,k):
    Ns = np.array(f.shape)
    Niprod = 1.0 / np.product(Ns)
    F = fft.fftn(f)
    
    p = max(0,min(100.0, 100.0 * (1.0 - float(k) / float(f.size))))
    Q = np.percentile(np.abs(F),p)

    half = Ns[0] / 2
    subF = F[:(half+1),...]
    it = np.nditer(subF, flags=['multi_index'])
    freq = []
    shift = []
    amp = []
    while not it.finished:
        if np.abs(F[it.multi_index]) < Q:
            it.iternext()
            continue

        coords = np.array(it.multi_index)
        R = np.real(F[it.multi_index])
        I = np.imag(F[it.multi_index])
        if np.abs(R) > 1e-12:
            #print 'Real', coords
            freq.append(2*np.pi*coords)
            shift.append(np.pi/2)
            if coords[0] == 0 or coords[0] == half:
                amp.append(R*Niprod)
            else:
                amp.append(2*R*Niprod)

            
        if np.abs(I) > 1e-12:
            #print 'Imag', coords
            freq.append(2*np.pi*coords)
            shift.append(0)
            if coords[0] == 0 or coords[0] == half:
                amp.append(-I*Niprod)
            else:
                amp.append(-2*I*Niprod)
        it.iternext()

    freq = np.vstack(freq)
    assert(freq.shape[1] == len(f.shape))
    shift = np.array(shift)
    amp = np.array(amp)
        
    return freq,shift,amp
    
class TrigBasis(Basis):
    def __init__(self,freq,shift):
        (M,d) = freq.shape
        (m,) = shift.shape
        assert(m==M)

        self.freq = freq.T
        self.shift = shift

        
    def evaluate(self,points,**kwargs):
        (N,D) = points.shape
        (d,M) = self.freq.shape
        assert(D == d)
        
        B = np.sin(points.dot(self.freq) + self.shift[np.newaxis,:])
        assert((N,M) == B.shape)
        return B

class TrigFunction(MultiFunction):
    def __init__(self,freq,shift,amps):
        (M,D) = freq.shape
        (m,K) = amps.shape
        assert(M == m)
        
        self.basis = TrigBasis(freq,shift)
        self.amps = amps

    def evaluate(self,points):
        B = self.basis.evaluate(points)
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
