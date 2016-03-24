import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from regular_grid import *
from irregular_grid import *
from indexer import *


#######################################
# AUX functions
    
def is_int(x):
    return x.dtype.kind in 'ui'

def make_points(gens,order='C'):
    """
    Makes the mesh in the order you would expect for
    np.reshape after.

    E.g. if handed [np.linspace(0,1,5),np.linspace(0,1,7)]
    then this would make the points P such that if mapped
    the 2D plot makes spacial sense. So np.reshape(np.sum(P),(5,7))
    would look pretty and not jumpy
    """
    if 'F' == order:
        gens = list(reversed(gens))
    if 1 == len(gens):
        return gens[0][:,np.newaxis] # meshgrid needs 2 args
    
    meshes = np.meshgrid(*gens,indexing='ij')
    points = np.column_stack([M.flatten() for M in meshes])
    if 'F' == order:
        return np.fliplr(points)
    return points    

#############################3
# Tests

def test_1d():    
    cuts = [[0,0.6,1]]
    grid_part = IrregularGridPartition(cuts)
    disc = IrregularGridDiscretizer(grid_part)

    N = 250
    samples = np.linspace(0,1,N)[:,np.newaxis]
    coords = disc.to_cell_coord(samples)
    assert((N,1) == coords.shape)

    for n in xrange(N):
        plt.plot([0,1],[samples[n,0],coords[n,0]],'-b',alpha=0.1)
    plt.show()

def test_index(order='C'):
    
    lens = [3,2,4,3]
    indexer = Indexer(lens,order=order)

    coords = make_points([xrange(x) for x in lens],order=order)
    indices = indexer.coords_to_indices(coords)
    backagain = indexer.indices_to_coords(indices)
    assert(np.all(coords == backagain))
    
    (N,) = indices.shape
    for n in xrange(N):
        print '{0} -> {1} -> {2}'.format(coords[n,:],
                                         indices[n],
                                         backagain[n,:])
def test_timing():
    N = 250
    M = N+1
    S = 10000
    grid_desc = [(0,1,N),(0,1,N)]
    cuts = IrregularGridPartition([np.linspace(0,1,N+1),
                                   np.linspace(0,1,N+1)])
    reg_grid = RegularGridPartition(grid_desc)
    reg_disc = RegularGridDiscretizer(reg_grid)
    irr_disc = IrregularGridDiscretizer(cuts)
    
    assert(reg_disc.num_cells == irr_disc.num_cells)

    points = make_points([np.linspace(0,1,M),np.linspace(0,1,M)])
    points = np.vstack([points,np.random.rand(S,2)])

    start = time.time()
    reg_coord = reg_disc.to_cell_coord(points)
    print 'Regular time:',time.time() - start
    start = time.time()
    irr_coord = irr_disc.to_cell_coord(points)
    print 'Irregular time:',time.time() - start

    error = np.sum(np.abs(reg_coord - irr_coord),axis=1)
    print 'Total error',np.sum(error[:]) 
    for n in xrange(M*M):
        if np.all(reg_coord[n,:] == irr_coord[n,:]):
            continue
        print '{0} -> {1},{2}'.format(points[n,:],
                                      reg_coord[n,:],
                                      irr_coord[n,:])
    assert(np.all(reg_coord == irr_coord))

def test_2d():
    N = 3
    S = 50
    cuts = IrregularGridPartition([np.linspace(0,1,N+1),
                                   np.linspace(0,1,N+1)])
    irr_disc = IrregularGridDiscretizer(cuts)
    
    points = make_points([np.linspace(0,1,S),
                          np.linspace(0,1,S)])
    coords = irr_disc.to_cell_coord(points)
    midpoints = irr_disc.midpoints(coords)
    (n,d) = midpoints.shape
    assert(d == 2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in xrange(n):
        ax.plot([points[i,0],midpoints[i,0]],
                    [points[i,1],midpoints[i,1]],
                    [0,1],'-b',alpha = 0.1)
    ax.view_init(elev=90,azim=0) # 2D view
    plt.show()
    
###############################
# Interp trial (should flesh out).
    
def trial_interpolate(X,low,high):
    (N,D) = X.shape
    assert(np.all(X >= 0))
    assert(np.all(X <= 1))
    assert((D,) == low.shape)
    assert((D,) == high.shape)
    
    Coef = np.empty((N,2*D))
    for d in xrange(D):
        Z = high[d] - low[d]
        Coef[:,2*d] = (X[:,d] - low[d])/Z
        Coef[:,2*d+1] = (high[d] - X[:,d])/Z

    Weights = np.empty((N,2**D))
    for i in xrange(2**D):
        b_str = np.binary_repr(i,width=D)
        b = np.array([int(x) for x in b_str],dtype=bool)
        vert = np.empty(2*D,dtype=bool)
        vert[::2] = b
        vert[1::2] = 1 - b
        Weights[:,i] = np.prod(Coef[:,vert],axis=1)
    return Weights
    
def test_interp():
    x = np.array([[0.2,0.1]])
    low = np.array([0,0])
    high = np.array([1,1])
    print trial_interpolate(x,low,high)

test_timing()
