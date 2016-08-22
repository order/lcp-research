import numpy as np
from rectilinear_grid import *
from rectilinear_indexer import *
from discretize import *

import matplotlib.pyplot as plt

def plot_pairwise(A,B):
    assert 2 == A.ndim
    assert np.all(A.shape == B.shape)

    (N,D) = A.shape
    for i in xrange(N):
        plt.plot([A[i,0],B[i,0]],[A[i,1],B[i,1]],'.-b',alpha=0.2)
    plt.show()

def plot_index_scatter(P,I,node_list):
    assert 2 == len(node_list)
    for (i,nl) in enumerate(node_list):
        l = node_list[1-i][0]
        u = node_list[1-i][-1]
        for cut in nl:            
            pts = np.array([[cut,cut],[l,u]])
            plt.plot(pts[i,:],pts[1-i,:],'-k',lw=2.0)
    plt.scatter(P[:,0],P[:,1],
                s=25,c=I,alpha=0.5,lw=0,cmap='prism')
    plt.show()

    
def sorted_random(N):
    points = np.concatenate([np.array([-1,1]), np.random.uniform(-1,1,N-2)])
    return np.sort(points)

#################
# INDEXER TESTS #
#################

def indexer_test_0():
    # Make sure that the fast indexer matches the slow indexer
    lens = np.array([5,6])
    indexer = Indexer(lens)

    for i in xrange(lens[0]):
        for j in xrange(lens[1]):
            coords = np.array([i,j])
            idx1 = indexer.coords_to_indices(coords[np.newaxis,:])[0]
            idx2 = slow_coord_to_index(coords,lens)
            assert(idx1 == idx2)    

######################
# REGULAR GRID TESTS #
######################
    
def regular_grid_test_0():
    """
    Map random physical points to cell index point
    Cell index point is always the least vertex in the cell,
    So 'error' should always be negative
    """
    
    grid_desc = [(-1,1,15),(-1,1,15)]
    RG = RegularGrid(grid_desc)

    points = np.random.uniform(-1,1,1500,2)
    indices = RG.points_to_indices(points)    
    recon_points = RG.indices_to_lowest_points(indices)
    
    #plot_pairwise(points,recon_points)

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
    """
    Plot mappings, visually scan for issues
    """
    grid_desc = [(-1,1,5),(-1,1,5)]
    RG = RegularGrid(grid_desc)

    # Make a little larger for NaN behavior
    points = np.random.uniform(-1.1,1.1,(5000,2))
    indices = RG.points_to_indices(points)

    node_list = [np.linspace(l,u,n+1) for (l,u,n) in grid_desc]
    plot_index_scatter(points,indices,node_list)

########################
# IRREGULAR GRID TESTS #
########################

def irregular_grid_test_0():
    """
    Map random physical points to cell index point
    Cell index point is always the least vertex in the cell,
    So 'error' should always be negative
    """

    node_list = [sorted_random(5),sorted_random(5)]
    
    IG = IrregularGrid(node_list)

    points = 2 * np.random.rand(1500,2) - 1
    indices = IG.points_to_indices(points)    
    recon_points = IG.indices_to_lowest_points(indices)
    
    plot_pairwise(points,recon_points)

    diff = points - recon_points
    assert np.all(diff >= 0)

def irregular_grid_test_1():
    """
    Map cell index points. Should be identity
    """
    node_list = [sorted_random(5),sorted_random(5)]

    IG = IrregularGrid(node_list)

    points = make_points([nl[:-1] for nl in node_list])
    indices = IG.points_to_indices(points)    
    recon_points = IG.indices_to_lowest_points(indices)

    plot_pairwise(points,recon_points)    
    
    diff = points - recon_points
    assert np.linalg.norm(diff) < 1e-9    

def irregular_grid_test_2():
    """
    Plot mappings, visually scan for issues
    """

    node_list = [sorted_random(5),sorted_random(5)]

    IG = IrregularGrid(node_list)

    # Make a little larger for NaN behavior
    points = np.random.uniform(-1.1,1.1,(5000,2))
    indices = IG.points_to_indices(points)

    plot_index_scatter(points,indices,node_list)


################################
# TABULAR DISCRETIZATION TESTS #
################################

def tabular_discretizer_test_0():
    grid_desc = [(-1,1,2),(-1,1,2)]
    RG = RegularGrid(grid_desc)

    points = 2 * np.random.rand(500,2) - 1
    disc = TabularDiscretizer(RG)
    dist = disc.locate(points)

    print dist.sum(axis=0)
    print dist.sum(axis=1)
    
    

#################
# MAIN FUNCTION #
#################
    
if __name__ == "__main__":

    indexer_test_0()
    
    #regular_grid_test_0()
    #regular_grid_test_1()
    #regular_grid_test_2()

    #irregular_grid_test_0()
    #irregular_grid_test_1()
    #irregular_grid_test_2()

    #tabular_discretizer_test_0()
