import numpy as np
import scipy.sparse as sps

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

        self.num_basis = grid.cell_indexer.get_num_nodes()
        self.num_physical_basis = grid.cell_indexer.get_num_physical_nodes()

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

def MultilinearInterpolation(object):
    def __init__(self,grid):
        self.grid = grid       

    def get_num_basis(self):
        """
        Gets the number of basis functions / nodes
        """
        raise NotImplementedError()

    def convert_to_sparse_matrix(self,indices,vertices,weights):
        (N,) = indices.shape
        D = self.grid.get_dim()

        indexer = self.grid.get_indexer()
        oob_mask = indexer.is_oob(indices)
        num_oob = np.sum(oob_mask)
        num_normal = N = num_oob
        normal_idx = np.arange(N)[~oob_mask]
        oob_idx = np.arange(N)[oob_mask]
        
        m = num_norm*(2**D) # Space for normal points
        M = m + num_oob # Add on space for oob nodes
        cols = np.empty(M)
        rows = np.empty(M)
        data = np.empty(M)

        # Add normal weights
        cols[:m] = (np.tile(normal_idx,(2**D,1)).T).flatten()
        rows[:m] = (vertices[~oob_mask,:]).flatten()
        data[:m] = (weights[~oob_mask,:]).flatten()

        # Route all oob points to oob node
        cols[B:] = oob_idx
        rows[B:] = indices[oob_idx]
        data[B:] = np.ones(num_oob)

        NN = self.grid.get_num_nodes()
        return sps.coo_matrix((data,(rows,cols)),shape=(NN,N))           
            
    def locate(self,points):
        (N,D) = points.shape
        assert D == self.grid.get_dim()

        # Get indices
        indices = points_to_indices(points) # Compute once
        
        dist = self.grid.get_rel_distance_to_low(points,indices)
        assert (N,D) == dist

        vertices = self.grid.get_neighbors(indices)
        assert (N,2**D) == vertices.shape
        
        # Calculate multilinear interp weights from distances
        weights = np.empty((N,2**D))
        for (i,diff) in enumerate(itertools.product([0,1],
                                                    repeat=D)):
            mask = np.array(diff,dtype=bool)
            weights[:,i] = np.product(dist[:,mask],axis=1)\
                           * np.product(1 - dist[:,~mask],axis=1)

        return convert_to_sparse_matrix(self,indices,vertices,weights)
