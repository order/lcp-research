import numpy as np
from grid import *
from indexer import *
from discretize import *
from coord import *
from utils import *

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

#############
# OOB TESTS #
#############
def oob_test_0():

    print "oob_test_0"
    grid_desc = [(-1,1,5),(-1,1,5)]
    RG = RegularGrid(grid_desc)
    points = np.array([[-1.01,0.9],
                       [1.01,0.9],
                       [0.9,-1.01],
                       [0.9,1.01]])
    oob = OutOfBounds()
    oob.build_from_points(RG,points)
    assert np.all(oob.indices == np.arange(4))


#################
# INDEXER TESTS #
#################

def indexer_test_0():
    print "indexer_test_0"
    
    # Make sure that the fast indexer matches the slow indexer
    grid_desc = [(-1,1,4),(-1,1,6)]
    (low,hi,num_cells) = zip(*grid_desc)
    num_cells = np.array(num_cells)
    RG = RegularGrid(grid_desc)

    for i in xrange(10):
        # Convert
        points = np.random.uniform(-1.0,1.0,(1,2))
        idx1 = RG.points_to_cell_indices(points)[0]
        # Double check
        coords = RG.points_to_cell_coords(points)
        raw_coords = coords.coords[0,:]
        idx2 = slow_coord_to_index(raw_coords,num_cells)
        idx3 = even_slower_coord_to_index(raw_coords,num_cells)
        assert idx1 == idx2
        assert idx1 == idx3

def indexer_test_1():
    """
    Reconstructing coord from indices the same as original convertion
    of points
    """
    print "indexer_test_1"
    
    grid_desc = [(-1,1,2),(-1,1,2)]
    RG = RegularGrid(grid_desc)

    # Make a little larger for NaN behavior
    points = np.random.uniform(-1.1,1.1,(10,2))
    coords = RG.points_to_cell_coords(points)
    indices = RG.points_to_cell_indices(points)

    recon_coords = RG.cell_indices_to_cell_coords(indices)
    
    assert np.all(coords.oob.mask == recon_coords.oob.mask)
    mask = coords.oob.mask
    assert np.all(coords.coords[~mask,:] == recon_coords.coords[~mask,:])

######################
# REGULAR GRID TESTS #
######################
    
def regular_grid_test_0():
    """
    Map random physical points to cell index point
    Cell index point is always the least vertex in the cell,
    So 'error' should always be negative
    """
    print "regular_grid_test_0"
    grid_desc = [(-1,1,7),(-1,1,4)]
    RG = RegularGrid(grid_desc)

    points = np.random.uniform(-1,1,(1000,2))
    indices = RG.points_to_cell_indices(points)
    recon_points = RG.cell_indices_to_low_points(indices)

    # Visually look at the mapping to the low points
    #plot_pairwise(points,recon_points)

    diff = points - recon_points
    assert np.all(diff >= 0)

def regular_grid_test_1():
    """
    Map cell index points. Should be identity
    """
    print "regular_grid_test_1"

    grid_desc = [(-1,1,15),(-1,1,25)]
    RG = RegularGrid(grid_desc)

    points = make_points([np.linspace(l,u,n+1)[:-1] for (l,u,n) in grid_desc])
    indices = RG.points_to_cell_indices(points)    
    recon_points = RG.cell_indices_to_low_points(indices)
    
    diff = points - recon_points
    assert np.linalg.norm(diff) < 1e-9    

def regular_grid_test_2():
    """
    Plot mappings, visually scan for issues
    """
    print "regular_grid_test_2"
    
    grid_desc = [(-1,1,4),(-1,1,9)]
    RG = RegularGrid(grid_desc)

    # Make a little larger for NaN behavior
    points = np.random.uniform(-1.1,1.1,(5000,2))
    indices = RG.points_to_cell_indices(points)

    node_list = [np.linspace(l,u,n+1) for (l,u,n) in grid_desc]
    plot_index_scatter(points,indices,node_list)

def regular_grid_test_3():
    """
    Some santity conditions on the cell coordinates
    """
    print "regular_grid_test_3"

    grid_desc = [(-1,1,3),(-1,1,16)]
    RG = RegularGrid(grid_desc)
    
    points = 1.25*np.random.uniform(-1,1,(5000,2))
    cell_coords = RG.points_to_cell_coords(points)

    oob = RG.are_points_oob(points)
    assert np.all(cell_coords.coords[~oob,:] < RG.num_cells[np.newaxis,:])
    assert np.all(cell_coords.coords[~oob,:] >= 0)

def regular_grid_test_4():
    """
    Checking oob behavior for regular grids
    """
    print "regular_grid_test_4"

    N = 3
    grid_desc = [(-1,1,N),(-1,1,N)]
    RG = RegularGrid(grid_desc)

    points = np.array([[-1.01,0.9],
                       [1.01,0.9],
                       [0.9,-1.01],
                       [0.9,1.01]])
    cell_coords = RG.points_to_cell_coords(points)
    vertices = RG.cell_coords_to_vertex_indices(cell_coords)
    expected = np.arange(4) + (N+1)**2
    assert np.all(vertices[:,0] == expected)
    
def regular_grid_test_5():
    """
    Map points to cell vertices
    """
    print "regular_grid_test_5"

    grid_desc = [(-1,1,4),(-1,1,5)]
    RG = RegularGrid(grid_desc)

    N = 125
    points = 1.25*np.random.uniform(-1,1,(N,2))
    cell_coords = RG.points_to_cell_coords(points)
    vertices = RG.cell_coords_to_vertex_indices(cell_coords)

    plt.plot(points[:,0],points[:,1],'.b')
    for v in xrange(4):
        V = RG.node_indices_to_node_points(vertices[:,v])
        for i in xrange(N):
            plt.plot([points[i,0],V[i,0]],
                     [points[i,1],V[i,1]],
                     '-b',alpha=0.25)
        plt.plot(V[:,0],V[:,1],'or')
    plt.title('Mapping random points to containing node')
    plt.show()

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
    """
    Some basic sanity tests for tabular discretization
    """
    print "tabular_discretizer_test_0"

    N = 5
    grid_desc = [(-1,1,N),(-1,1,N)]
    RG = RegularGrid(grid_desc)

    disc = TabularDiscretizer(RG)
    assert (N*N + 4) == disc.get_num_basis()

    points = np.random.uniform(-1.1,1.1,(1500,2))
    dist = disc.points_to_basis_dists(points)

    num_phys = disc.num_spatial_basis-1
    assert num_phys == N*N-1

    # Covers physical spaces
    assert np.all((dist.sum(axis=1))[:num_phys] > 0)

    # Is stochastic
    assert np.max(np.abs(dist.sum(axis=0) - 1)) < 1e-8

def tabular_discretizer_test_1():
    """
    Interpolate function
    """
    print "tabular_discretizer_test_1"

    grid_desc = [(-1,1,4),(-1,1,6)]
    RG = RegularGrid(grid_desc)

    disc = TabularDiscretizer(RG)
    N = disc.get_num_basis()
    f = np.arange(N)
    
    points = np.random.uniform(-1.1,1.1,(5000,2))
    dist = disc.points_to_basis_dists(points)

    g = dist.T.dot(f)

    plt.scatter(points[:,0],
                points[:,1],
                c=g,s=25,cmap='prism')
    plt.title('Interpolation')
    plt.show()


####################################
# MULTILINEAR DISCRETIZATION TESTS #
####################################

def multilinear_discretizer_test_0():
    """
    Some basic sanity tests for multilinear discretization
    """
    print "multilinear_discretizer_test_0"

    N = 5
    grid_desc = [(-1,1,N),(-1,1,N)]
    RG = RegularGrid(grid_desc)

    disc = MultilinearInterpolation(RG)
    assert ((N+1)*(N+1) + 4) == disc.get_num_basis()

    points = np.random.uniform(-1.1,1.1,(8,2))
    dist = disc.points_to_basis_dists(points)

    num_phys = disc.num_spatial_basis-1
    assert num_phys == N*N-1

    # Covers physical spaces
    assert np.all((dist.sum(axis=1))[:num_phys] > 0)

    # Is stochastic
    assert np.max(np.abs(dist.sum(axis=0) - 1)) < 1e-8

#################
# MAIN FUNCTION #
#################
    
if __name__ == "__main__":

    oob_test_0()
    indexer_test_0()
    indexer_test_1()
    
    regular_grid_test_0()
    regular_grid_test_1()
    #regular_grid_test_2() # Visually inspect plots
    regular_grid_test_3()
    regular_grid_test_4()
    #regular_grid_test_5() # Visually inspect plots

    #irregular_grid_test_0()
    #irregular_grid_test_1()
    #irregular_grid_test_2()

    tabular_discretizer_test_0()
    #tabular_discretizer_test_1() # Visually inspect plots
    
    multilinear_discretizer_test_0()
