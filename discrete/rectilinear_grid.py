import numpy as np
import scipy.sparse as sps

from rectilinear_indexer import Indexer

from utils import is_sorted

##############################
# ABSTRACT RECTILINEAR GRID #
##############################
class Grid(object):
    # Main function: convert points to cell coordinates
    def points_to_cell_coords(self,points):
        raise NotImplementedError()
    def points_to_indices(self,points):
        raise NotImplementedError()       
    def indices_to_lowest_points(self,coords):
        raise NotImplementedError()
    
    # Overall space is an axis-aligned rectangle
    def get_lower_boundary(self):
        return self.lower_bound    
    def get_upper_boundary(self):
        return self.upper_bound
    def get_dim(self):
        return self.dim

    def get_indexer(self):
        return indexer

    def get_num_nodes(self):
        # Total number (scalar), include oob
        return self.indexer.max_index+1
    
    def get_num_real_nodes(self):
        # Exclude oob
        return self.indexer.physical_max_index+1
    
    def get_num_oob(self):
        return self.num_nodes() - self.num_real_nodes()
    
    def get_oob_range(self):
        return xrange(self.num_real_nodes(),self.num_nodes())


##############################
# REGULAR RECTILINEAR GRID #
##############################
class RegularGrid(Grid):
    def __init__(self,grid_desc):
        self.dim = len(grid_desc)
        self.grid_desc = grid_desc # List of (low,high,num) triples

        # Number of cutpoints along each dimension
        (low,hi,num) = zip(*self.grid_desc)
        self.lower_bound = np.array(low,dtype=np.double)
        self.upper_bound = np.array(hi,dtype=np.double)
        self.num_cells = np.array(num)
        
        assert not np.any(self.num_cells == 0)
        self.num_nodes = self.num_cells + 1

        self.delta = (self.upper_bound - self.lower_bound)
        self.delta /= self.num_cells.astype(np.double)

        # Initialize the indexer
        self.indexer = Indexer(self.num_nodes)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-12        

    def points_to_cell_coords(self,points):
        """
        Figure out which cell each point is in
        """
        (N,D) = points.shape
        assert D == self.dim
        
        Coords = np.empty((N,D))
        for d in xrange(D):
            (low,high,n) = self.grid_desc[d]
            hi_cell = n-1
            # Transform: [low,high) |-> [0,n)
            transform = n * (points[:,d] - low) / (high - low) + self.fuzz
            Coords[:,d] = np.floor(transform)

            # Fuzz top boundary to get [low,high]
            fuzz_mask = np.logical_and(high <= points[:,d],
                                     points[:,d] < high + 2*self.fuzz)
            Coords[fuzz_mask,d] = hi_cell
        return Coords
    
    def points_to_indices(self,points):
        coords = self.points_to_cell_coords(points)
        return self.indexer.coords_to_indices(coords)

    def indices_to_lowest_points(self,indices):
        assert 1 == indices.ndim
        
        coords = self.indexer.indices_to_coords(indices)
        return self.coords_to_lowest_points(coords)
        
    def coords_to_lowest_points(self,coords):
        assert 2 == coords.ndim
        (N,D) = coords.shape
        assert self.dim == D

        points = self.lower_bound + coords * self.delta
        assert (N,D) == points.shape

        return points

##############################
# IRREGULAR RECTILINEAR GRID #
##############################
class IrregularGrid(Grid):
    """
    Rectilinear grid from irregular, but sorted, list of node locations
    """
    def __init__(self,node_lists):
        self.dim = len(node_lists)
        self.node_lists = np.array(node_lists)
        # List of np.ndarray cutpoint locations

        for nl in node_lists:
            assert nl.ndim == 1 # 1D array
            assert nl.size >= 2 # At least two nodes
            assert is_sorted(nl)
        
        # Number of cutpoints along each dimension
        desc = [(nl[0],nl[-1],nl.size) for nl in node_lists]
        (low,hi,num) = zip(*desc)
        self.lower_bound = np.array(low)
        self.upper_bound = np.array(hi)
        self.num_nodes = np.array(num)
        
        self.num_cells = self.num_nodes - 1

        # Initialize the indexer
        self.indexer = Indexer(self.num_nodes)

        # Fuzz to convert [low,high) to [low,high]
        self.fuzz = 1e-12
    
    def points_to_cell_coords(self,points):
        (N,D) = points.shape
        assert D == self.dim

        Coords = np.empty((N,D))
        for d in xrange(D):
            # Find the correct position in the dth node list
            coord = np.searchsorted(self.node_lists[d],
                                    points[:,d],
                                    side='right') - 1
            # The 'right' is important if points are exactly on the node
            assert (N,) == coord.shape
            Coords[:,d] = coord

            # Include the upper boundary
            ub = self.upper_bound[d]
            hi_cell = self.num_cells[d] - 1
            fuzz_mask = np.logical_and(points[:,d] >= ub,
                                       points[:,d] < ub + self.fuzz)
            Coords[fuzz_mask,d] = hi_cell

            # Indexer will take care of mapping to correct OOB node
            #lb = self.lower_bound[d]
            #oob_mask = np.logical_or(points[:,d] < lb,
            #                         points[:,d] >= ub+self.fuzz)
            #Coords[oob_mask,d] = np.nan
        return Coords

    def points_to_indices(self,points):
        coords = self.points_to_cell_coords(points)
        return self.indexer.coords_to_indices(coords)

    def indices_to_lowest_points(self,indices):
        assert 1 == indices.ndim
        
        coords = self.indexer.indices_to_coords(indices)
        return self.coords_to_lowest_points(coords)
        
    def coords_to_lowest_points(self,coords):
        assert 2 == coords.ndim
        (N,D) = coords.shape
        assert self.dim == D

        points = np.empty((N,D))
        for d in xrange(D):
            points[:,d] = self.node_lists[d,coords[:,d]]

        return points

