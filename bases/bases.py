import math
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import lcp.util
import itertools
from collections import defaultdict

import time

class BasisGenerator(object):
    """
    This is an object that wraps around a "basic" basis generator and
    deals with special nodes like out of bounds nodes.

    The assumption is that a special node requires a distinct elementary
    vector associated with it because it is sufficiently different
    """
    def __init__(self,gen_obj):
        self.generator_function = gen_obj
        self.remappers = None
        """
        TODO: implement "remapper" logic so that special nodes can be mapped
        to real physical locations that behave like we'd expect the special 
        node to be similar to
        """        

    def generate(self, points,K,**kwargs):
        """
        Generates K basis columns based on the provided points.
        
        Note that any special points (e.g. non-physical points) will 
        be given an elementary column vector. 
        """

        (N,D) = points.shape

        # Special points are things like OOB nodes    
        special_points = sorted(kwargs.get('special_points',[]))
        sp_set = set(special_points)
        S = len(sp_set)         

        # Make sure any NaN row that wasn't remapped is a special state.
        nan_mask = np.any(np.isnan(points),axis=1)
        assert((N,) == nan_mask.shape)
        nan_set = set(nan_mask.nonzero()[0])            
        assert(nan_set <= sp_set)
        # All non-phyiscal states should be special
           
        assert(K >= S) # Should have at least this many bases

        # Generate all the bases, reserving bases for the special 
        B = self.generator_function.generate(points,K - S)

        # 
        B[special_points,:] = 0 # Blot all special points out
        B = np.hstack([B,np.zeros((N,S))])
        B[(K-S):,special_points] = np.eye(S)

        return B

class BasicBasisGenerator(object):
    """
    The BasisGenerator uses these for actually doing to mapping of rows to basis values.
    It handles some of the higher-level concerns like cleaning and special point handling
    """
    def __init__(self):
        pass
    def generate(self,points,K,**kwargs):
        raise NotImplementedError()

class RandomFourierBasis(BasicBasisGenerator):
    def __init__(self,**kwargs):
        pass
    def generate(self,points,K,**kwargs):
        assert(K >= 1)
        (N,D) = points.shape

        # Format: B = [1 | fourier columns]
        W = np.random.randn(D,K-1) # Weights
        Phi = 2.0 * np.pi * np.random.rand(K-1) # Phases shift
        F = np.sin(points.dot(W) + Phi) # Non-constant columns
        assert((N,K-1) == F.shape)
        B = np.hstack([np.ones((N,1)),F])
        
        assert((N,K) == B.shape)
        return B

def random_fourier_test():
    N = 50
    grid = np.linspace(0,2*np.pi,N)
    Meshes = np.meshgrid(grid,grid,order='ij')
    Points = np.column_stack([x.flatten() for x in Meshes])

    B = random_fourier(Points,1)
    Img = B.reshape((N,N))
    plt.imshow(Img,interpolation=None)
    plt.show()

def projection_test():
    n = 25
    N = n**2
    k = n

    grid = np.linspace(0,2*np.pi,n)
    Meshes = np.meshgrid(grid,grid,order='ij')
    Points = np.column_stack([x.flatten() for x in Meshes])

    B = random_fourier(Points,k)
    Z = np.zeros(B.shape)
    BB = np.vstack([np.hstack([B,Z]),np.hstack([Z,B])])
    B_pinv = np.linalg.pinv(B)
    print B.shape,B_pinv.shape
    BB_pinv = np.linalg.pinv(BB)
    BB_pinv2 = np.vstack([np.hstack([B_pinv,Z.T]),np.hstack([Z.T,B_pinv])])
   
    print 'rank(B)', np.linalg.matrix_rank(B)
    print 'norm(BB_pinv - BB_pinv2)', np.linalg.norm(BB_pinv - BB_pinv2)

    Pi = B.dot(B_pinv)
    print 'norm(Pi):', np.linalg.norm(Pi,ord=2)
    print 'norm(Pi**2 - Pi)', np.linalg.norm(Pi.dot(Pi) - Pi)

    #plt.imshow(Pi,interpolation=None)
    #plt.show()   

def fourier(N,k):
    """
    Creates a real DFT-style matrix of size N x (2k-1)
    The first half is the cosine component in descending order
    Then the middle column is all one,
    Then the second half is the sine component in ascending order
    
    Note that this is the 1D "DFT" so may have kind of strange behaviour
    on "flattened" multidimensional vectors
    """
    x = np.linspace(0,2*np.pi, num=N, endpoint=False)
    w = np.array(range(k)) # Frequencies 
    C = math.sqrt(2) / math.sqrt(N) * np.cos(np.outer(x,w))
    C[:,0] /= math.sqrt(2)
    S = math.sqrt(2) / math.sqrt(N) * np.sin(np.outer(x,w[1:]))
    return np.hstack([np.fliplr(C),S])

def chebyshev(N,k):
    if k == 1:
        return np.ones(N,1)

    x = np.linspace(-1,1,num=N,endpoint=False)
    B = [np.ones(N),x]
        
    while k > 2:
        T = 2*x*B[-1] - B[-2]
        B.append(T)
        k -= 1
    return np.column_stack(B)

def cmac(shape, grids):
    """
    Creates a CMAC (Cerebellar model articulation controller) basis.
    The shape is the number of uniform cells in the state space
    Each grid is defined by an offset and the extent of a single cell.
    
    Returns a sparse matrix
    """
    D = len(shape)
    Bases = []
    for (offset,extent) in grids:
        assert(len(offset) == D)
        assert(len(extent) == D)
        
        # Assign cells to blocks in grid
        cells = defaultdict(list)
        I = 0
        for (id,coords) in enumerate(itertools.product(*[xrange(x) for x in shape])):
            block_coords =  [int(coords[i] - offset[i]) / int(extent[i])\
                for i in xrange(D)]
            cells[tuple(block_coords)].append(id)
            I = id
           
        #
        B = scipy.sparse.dok_matrix((I+1,len(cells))) # TODO: is DOK the best? Better way to construct?
        for (block_id,(block_coord,cell_list)) in enumerate(cells.items()):
            for cell_id in cell_list:
                B[cell_id,block_id] = 1.0
        Bases.append(B)
    
    return scipy.sparse.hstack(Bases).tocsr()
    
def cmac_test():
    """
    Example of function approximation using CMACs
    Approximates a Gaussian via least-squares
    """
    shape = (100,100)
    grid1 = [(0,0),(8,1)]
    grid2 = [(1,0),(5,10)]
    grids = [grid1,grid2]
    
    start = time.time()
    B = cmac(shape,grids)
    print 'Time',time.time() - start
    
    X,Y = np.meshgrid(*[np.linspace(0,1,shape[i]) for i in xrange(len(shape))])
    P = np.hstack([X.reshape(-1,1),Y.reshape(-1,1)])
    f = np.exp(-np.linalg.norm(P - 0.5,axis=1) ** 2)
    w = scipy.sparse.linalg.lsqr(B,f)[0]
    Z = (B.dot(w)).reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,rstride=1, cstride=1)
    #axes[2].imshow(B,interpolation='nearest')
    plt.show()
    
def rbf():
    """
    stub for radial basis function basis
    """
    pass
    
def fourier_test():
    N = 500
    F = fourier(N,25)
    x = np.array(range(N))
    f = np.exp(-(x - N/2) ** 2 / 1000.0)
    w = np.linalg.lstsq(F,f)[0]
    plt.plot(x,f,x,F.dot(w))
    plt.show()
