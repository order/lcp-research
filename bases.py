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

def fourier(N,k):
    """
    Creates a real DFT-style matrix of size N x (2k-1)
    The first half is the cosine component in descending order
    Then the middle column is all one,
    Then the second half is the sine component in ascending order
    
    Note that this is the 1D "DFT" so may have kind of strange behaviour
    on "flattened" multidimensional vectors
    """
    x = np.array(range(N))
    w = np.array(range(k)) / float(N) # Frequencies 
    C = np.cos(np.outer(x,w))
    S = np.sin(np.outer(x,w[1:]))
    return np.hstack([np.fliplr(C),S])    

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
    
def fourier_test():
    N = 500
    F = fourier(N,25)
    x = np.array(range(N))
    f = np.exp(-(x - N/2) ** 2 / 1000.0)
    w = np.linalg.lstsq(F,f)[0]
    plt.plot(x,f,x,F.dot(w))
    plt.show()
