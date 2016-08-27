import numpy as np
import scipy.sparse as sps

from utils import is_mat,is_vect,is_int,is_float,banner

from coord import Coordinates
import itertools

class Discretizer(object):
    
    def points_to_basis_dists(self,points):
        """
        Takes in (n,d) points,
        Finds a distributions over basis functions
        """
        raise NotImplementedError()

    def get_num_basis(self):
        """
        Gets the number of basis functions / nodes
        """
        raise NotImplementedError()
        

#######################
# TABULAR DISCRETIZER #
#######################

class TabularDiscretizer(Discretizer):
    def __init__(self,grid):
        self.grid = grid
        self.dim = grid.dim

        # Each cell gets a basis function
        self.num_basis = grid.cell_indexer.get_num_nodes()
        self.num_spatial_basis = grid.cell_indexer.get_num_spatial_nodes()

    def get_num_basis(self):
        return self.num_basis

    def points_to_basis_dists(self,points):
        """
        The location distribution for a point is the 
        index of the cell that it falls in.
        o - o - o
        | 2 | 3 |
        o - o - o
        | 0 | 1 |
        o - o - o
        
        """
        (N,D) = points.shape
        assert (D == self.grid.get_dim())

        cell_indices = self.grid.points_to_cell_indices(points)
        assert(not np.any(np.isnan(cell_indices)))

        cols = np.arange(N)
        rows = cell_indices
        data = np.ones(N,dtype=np.double)

        M = self.num_basis
        assert np.all(cell_indices < M)
        assert np.all(cell_indices >= 0)
        return sps.coo_matrix((data,(rows,cols)),shape=(M,N))

#############################
# MULTILINEAR INTERPOLATION #
#############################

class MultilinearInterpolation(Discretizer):
    def __init__(self,grid):
        self.grid = grid
        self.dim = grid.dim

        # Each node gets a basic function
        self.num_basis = grid.node_indexer.get_num_nodes()
        self.num_spatial_basis = grid.node_indexer.get_num_spatial_nodes()

    def get_num_basis(self):
        return self.num_basis

    def convert_to_sparse_matrix(self,cell_coords,vertices,weights):
        assert isinstance(cell_coords,Coordinates)
        assert cell_coords.check()
        assert is_mat(vertices)
        assert is_int(vertices)
        assert is_mat(weights)
        assert is_float(weights)
        
        (N,D) = cell_coords.shape
        assert vertices.shape == weights.shape
        assert (N,2**D) == vertices.shape
        assert D == self.dim

        oob_mask = cell_coords.oob.mask
        num_oob = cell_coords.oob.num_oob()
        num_normal = N - num_oob
        assert num_oob >= 0
        assert num_normal >= 0
        
        normal_idx = np.arange(N)[~oob_mask]
        oob_idx = np.arange(N)[oob_mask]
        
        m = num_normal*(2**D) # Space for normal points
        M = m + num_oob # Add on space for oob nodes
        cols = np.empty(M)
        rows = np.empty(M)
        data = np.empty(M)

        # Add normal weights
        cols[:m] = (np.tile(normal_idx,(2**D,1)).T).flatten()
        rows[:m] = (vertices[~oob_mask,:]).flatten()
        data[:m] = (weights[~oob_mask,:]).flatten()

        # Route all oob points to oob node
        cols[m:] = oob_idx
        rows[m:] = vertices[oob_mask,0]
        data[m:] = np.ones(num_oob)

        NN = self.grid.get_num_total_nodes()
        point_dist = sps.coo_matrix((data,(rows,cols)),shape=(NN,N))
        point_dist = point_dist.tocsr()
        point_dist.eliminate_zeros()
        return point_dist
            
    def points_to_basis_dists(self,points):
        assert is_mat(points)
        assert is_float(points)
        (N,D) = points.shape
        assert D == self.grid.get_dim()

        G = self.grid

        # Get indices
        cell_coords = G.points_to_cell_coords(points)

        # Get rel distances
        rel_dist = G.points_to_low_vertex_rel_distance(points,
                                                       cell_coords)
        assert (N,D) == rel_dist.shape

        # Get the vertices
        vertices = self.grid.cell_coords_to_vertex_indices(cell_coords)
        assert (N,2**D) == vertices.shape
        
        # Calculate multilinear interp weights from distances
        weights = np.empty((N,2**D))
        for (i,bin_vertex) in enumerate(itertools.product([0,1],
                                                    repeat=D)):
            vert_mask = np.array(bin_vertex,dtype=bool)
            weights[:,i] = np.product(rel_dist[:,vert_mask],axis=1)\
                           * np.product(1.0 - rel_dist[:,~vert_mask],axis=1)

        point_dist = self.convert_to_sparse_matrix(cell_coords,
                                                   vertices,
                                                   weights)
        return point_dist
