import numpy as np
from rectilinear_grid import Grid,RegularGrid,IrregularGrid
from discretize import make_points, TabularDiscretizer

import matplotlib.pyplot as plt

def plot_pairwise(A,B):
    assert 2 == A.ndim
    assert np.all(A.shape == B.shape)

    (N,D) = A.shape
    for i in xrange(N):
        plt.plot([A[i,0],B[i,0]],[A[i,1],B[i,1]],'.-b',alpha=0.2)
    plt.show()
    
def regular_grid_test_0():
    """
    Map random points to cell index point
    Cell index point is always the least vertex in the cell,
    So 'error' should always be negative
    """
    
    grid_desc = [(-1,1,15),(-1,1,15)]
    RG = RegularGrid(grid_desc)

    points = 2 * np.random.rand(1500,2) - 1
    indices = RG.points_to_indices(points)    
    recon_points = RG.indices_to_lowest_points(indices)
    
    plot_pairwise(points,recon_points)

    diff = points - recon_points
    assert np.all(diff >= 0)

def regular_grid_test_1():
    """
    Map cell index points. Should be identity
    """
    grid_desc = [(-1,1,15),(-1,1,25)]
    RG = RegularGrid(grid_desc)

    points = make_points([np.linspace(l,u,n+1)[:-1] for (l,u,n) in grid_desc])
    indices = RG.points_to_indices(points)    
    recon_points = RG.indices_to_lowest_points(indices)
    
    diff = points - recon_points
    assert np.linalg.norm(diff) < 1e-9    

def regular_grid_test_2():
    grid_desc = [(-1,1,5),(-1,1,5)]
    RG = RegularGrid(grid_desc)

    points = 2 * np.random.rand(2500,2) - 1
    indices = RG.points_to_indices(points)    

    plt.scatter(points[:,0],points[:,1],
                s=25,c=indices,alpha=0.5,lw=0,cmap='plasma')
    plt.show()

def tabular_discretizer_test_0():
    grid_desc = [(-1,1,5),(-1,1,5)]
    RG = RegularGrid(grid_desc)

    points = 2 * np.random.rand(500,2) - 1
    disc = TabularDiscretizer(RG)
    dist = disc.locate(points)

    (N,D) = points.shape
    assert N == dist.shape[1]
    assert np.linalg.norm(dist.sum(axis=0) - np.ones(N)) < 1e-12
    
if __name__ == "__main__":
    #regular_grid_test_0()
    #regular_grid_test_1()
    #regular_grid_test_2()

    tabular_discretizer_test_0()
